import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()

        # =====================
        # Basic settings
        # =====================
        self.ratio = args.ratio
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.h = args.horizon

        # =====================
        # Deep backbone
        # =====================
        self.hidR = args.hidRNN
        self.GRU1 = nn.GRU(self.m, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)

        # =====================
        # Residual module
        # =====================
        self.residual_window = args.residual_window
        if self.residual_window > 0:
            self.residual_window = min(self.residual_window, self.P)
            self.residual = nn.Linear(self.residual_window, 1)

        # =====================
        # Output activation
        # =====================
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        elif args.output_fun == 'tanh':
            self.output = torch.tanh

        # =====================
        # Adjacency & mobility
        # =====================
        self.mask_mat = nn.Parameter(torch.Tensor(self.m, self.m))
        nn.init.xavier_uniform_(self.mask_mat)
        self.adj = data.adj

        # =====================
        # Epidemiological GRUs
        # =====================
        self.GRU_beta = nn.GRU(1, self.hidR, batch_first=True)
        self.GRU_gamma = nn.GRU(1, self.hidR, batch_first=True)
        self.GRU_sigma = nn.GRU(1, self.hidR, batch_first=True)

        self.PredBeta = nn.Sequential(
            nn.Linear(self.hidR, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

        self.PredGamma = nn.Sequential(
            nn.Linear(self.hidR, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

        self.PredSigma = nn.Sequential(
            nn.Linear(self.hidR, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

        # =====================
        # SEIR latent state head
        # =====================
        self.SEIR_head = nn.Sequential(
            nn.Linear(self.hidR, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # S, E, I, R
        )

    # ==================================================
    # Forward
    # ==================================================
    def forward(self, x):
        # x: [batch, window, regions]
        b = x.shape[0]
        xOriginal = x.clone()

        # =====================
        # Learn mobility matrix
        # =====================
        masked_adj = self.adj * F.softmax(self.mask_mat, dim=1)

        # =====================
        # Epidemiological parameters
        # =====================
        x_epi = x.permute(0, 2, 1).contiguous().view(-1, self.P, 1)

        beta_h, _ = self.GRU_beta(x_epi)
        gamma_h, _ = self.GRU_gamma(x_epi)
        sigma_h, _ = self.GRU_sigma(x_epi)

        Beta = self.PredBeta(beta_h[:, -1]).view(b, self.m)
        Gamma = self.PredGamma(gamma_h[:, -1]).view(b, self.m)
        Sigma = self.PredSigma(sigma_h[:, -1]).view(b, self.m)

        # =====================
        # NGM construction
        # =====================
        BetaDiag = torch.diag_embed(Beta)
        GammaDiag = torch.diag_embed(Gamma)

        diag_adj = torch.diag(torch.diagonal(masked_adj))
        W = torch.diag(torch.sum(masked_adj, dim=0)) - diag_adj
        A = ((masked_adj.T - diag_adj) - W).repeat(b, 1).view(b, self.m, self.m)

        NGMatrix = BetaDiag.bmm((GammaDiag - A).inverse())
        EpiOutput = xOriginal[:, -1, :].unsqueeze(1).bmm(NGMatrix).squeeze(1)

        # =====================
        # Deep spatiotemporal path
        # =====================
        xTrans = x.matmul(masked_adj)
        r = xTrans.permute(1, 0, 2)
        _, r = self.GRU1(r)
        r = self.dropout(r.squeeze(0))

        res = self.linear1(r)

        if self.residual_window > 0:
            z = x[:, -self.residual_window:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.residual_window)
            z = self.residual(z).view(b, self.m)
            res = res * self.ratio + z

        if self.output is not None:
            res = self.output(res)

        # =====================
        # SEIR latent states
        # =====================
        SEIR = self.SEIR_head(r)  # [b, 4]

        S = SEIR[:, 0]
        E = SEIR[:, 1]
        I = SEIR[:, 2]
        R = SEIR[:, 3]

        # Normalize per sample
        N = S + E + I + R + 1e-6
        S, E, I, R = S/N, E/N, I/N, R/N

        # Broadcast to regions
        S = S.unsqueeze(1).expand(-1, self.m)
        E = E.unsqueeze(1).expand(-1, self.m)
        I = I.unsqueeze(1).expand(-1, self.m)
        R = R.unsqueeze(1).expand(-1, self.m)


        # =====================
        # Physics-Informed SEIR residual
        # =====================
        dS = -Beta * S * I
        dE = Beta * S * I - Sigma * E
        dI = Sigma * E - Gamma * I
        dR = Gamma * I

        physics_residual = (
            dS.pow(2).mean() +
            dE.pow(2).mean() +
            dI.pow(2).mean() +
            dR.pow(2).mean()
        )

        return (
            res,               # Deep forecast
            EpiOutput,         # NGM forecast
            Beta,
            Sigma,
            Gamma,
            NGMatrix,
            physics_residual   # PINN loss
        )
