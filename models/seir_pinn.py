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
            nn.Linear(self.hidR + 1, 64),
            nn.Tanh(),
            nn.Linear(64, self.m * 4)
        )


    # ==================================================
    # Forward
    # ==================================================
    def forward(self, x, t):
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
        # Time variable (PINN)
        # =====================
        t = t.requires_grad_(True)   # t: [b, 1]

        # =====================
        # SEIR latent states (per region)
        # =====================
        rt = torch.cat([r, t], dim=1)               # [b, hidR + 1]
        SEIR = self.SEIR_head(rt).view(b, self.m, 4)

        S = SEIR[:, :, 0]
        E = SEIR[:, :, 1]
        I = SEIR[:, :, 2]
        R = SEIR[:, :, 3]

        # Normalize per region
        N = S + E + I + R + 1e-6
        S, E, I, R = S/N, E/N, I/N, R/N

        # =====================
        # Global coupling via mobility
        # =====================
        I_eff = I.matmul(masked_adj)   # [b, m]

        # =====================
        # Time derivatives via autograd
        # =====================

        dS_dt = torch.autograd.grad(
            S,
            t,
            grad_outputs=torch.ones_like(S),
            create_graph=True
        )[0]

        dE_dt = torch.autograd.grad(
            E,
            t,
            grad_outputs=torch.ones_like(E),
            create_graph=True
        )[0]

        dI_dt = torch.autograd.grad(
            I,
            t,
            grad_outputs=torch.ones_like(I),
            create_graph=True
        )[0]

        dR_dt = torch.autograd.grad(
            R,
            t,
            grad_outputs=torch.ones_like(R),
            create_graph=True
        )[0]

        # =====================
        # SEIR ODE definitions
        # =====================
        fS = -Beta * S * I_eff
        fE =  Beta * S * I_eff - Sigma * E
        fI =  Sigma * E - Gamma * I
        fR =  Gamma * I

        # =====================
        # Physics-informed residual (PINN)
        # =====================
        physics_residual = (
            (dS_dt - fS).pow(2).mean() +
            (dE_dt - fE).pow(2).mean() +
            (dI_dt - fI).pow(2).mean() +
            (dR_dt - fR).pow(2).mean()
        )

        # =====================
        # SEIR forward integration (Euler)
        # =====================
        S_t, E_t, I_t, R_t = S, E, I, R
        I_preds = []

        for _ in range(self.h):
            I_eff = I_t.matmul(masked_adj)

            dS = -Beta * S_t * I_eff
            dE =  Beta * S_t * I_eff - Sigma * E_t
            dI =  Sigma * E_t - Gamma * I_t
            dR =  Gamma * I_t

            S_t = S_t + dS
            E_t = E_t + dE
            I_t = I_t + dI
            R_t = R_t + dR

            # =====================
            # Issue 3 fix: population conservation
            # =====================
            N = S_t + E_t + I_t + R_t + 1e-6
            S_t = S_t / N
            E_t = E_t / N
            I_t = I_t / N
            R_t = R_t / N

            I_preds.append(I_t)


        I_forecast = torch.stack(I_preds, dim=1)  # [b, horizon, m]

        return (
            res,              # Deep forecast
            EpiOutput,        # NGM forecast
            I_forecast,       # SEIR-PINN forecast
            Beta,
            Sigma,
            Gamma,
            NGMatrix,
            physics_residual  # True PINN loss
        )
