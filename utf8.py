# adaptive_morphism_maxwell.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utilitaires : grilles & TV
# ---------------------------------------------------------------------------

def make_base_grid(H, W, device):
    """Grille normalisée pour grid_sample: shape (1,H,W,2) dans [-1,1]."""
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij"
    )
    grid = torch.stack([xs, ys], dim=-1).unsqueeze(0)  # (1,H,W,2)
    return grid

def tv_norm(flow, eps=1e-6):
    """
    Total Variation isotropique sur un champ (N,H,W,2).
    Renvoie la TV moyenne par échantillon.
    """
    dx = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dy = flow[:, 1:, :, :] - flow[:, :-1, :, :]
    tv = torch.sqrt(dx.pow(2).sum(-1) + dy.pow(2).sum(-1) + eps)
    return tv.mean()

# ---------------------------------------------------------------------------
# Fiabilité Maxwellienne à partir de stats regard
# ---------------------------------------------------------------------------

class MaxwellReliability(nn.Module):
    """
    Convertit des features de regard en (D_i, T_i, r_i).
    Entrée gaze_feats par lettre: [dur, disp, reg, v_mean, v_var, fix_ratio]
      - dur: durée moyenne de fixation (s)
      - disp: dispersion spatiale (px normalisés)
      - reg: taux de régressions (0-1)
      - v_mean: vitesse moyenne saccadique
      - v_var: variance de la vitesse (=> température T)
      - fix_ratio: ratio de temps en fixation
    """
    def _init_(self, alpha=0.6, beta=6.0, c=0.6):
        super()._init_()
        self.alpha = alpha
        self.beta = beta
        self.c = c

        # Normalisations apprenables (robuste aux unités hétérogènes)
        self.norm = nn.Parameter(torch.tensor([0.6, 0.4, 0.3, 50.0, 200.0, 0.8]), requires_grad=False)

        # Poids pour difficulté intrinsèque
        self.wD = nn.Parameter(torch.tensor([0.45, 0.25, 0.30, 0.0, 0.0, 0.1]), requires_grad=True)

    def forward(self, gaze_feats):
        """
        gaze_feats: (B,L,6)
        Retourne:
          D: (B,L) difficulté normalisée 0–1
          T: (B,L) température (∝ v_var)
          r: (B,L) fiabilité (0–1)
        """
        x = gaze_feats / (self.norm + 1e-9)
        dur, disp, reg, v_mean, v_var, fix_ratio = torch.unbind(x, dim=-1)
        # difficulté
        D = torch.clamp((self.wD[0]*dur + self.wD[1]*disp + self.wD[2]*reg + self.wD[5]*fix_ratio), 0.0, 1.5)
        D = torch.clamp(D, 0.0, 1.0)
        # température ~ variance de vitesse + dispersion
        T = torch.clamp(0.7*v_var + 0.3*disp, 0.0, 2.0)
        # fiabilité: plus T est grand (+ dispersion), plus r baisse
        r = torch.sigmoid(self.beta*(self.c - (self.alpha*T + (1.0-self.alpha)*disp)))
        return D, T, r

# ---------------------------------------------------------------------------
# Modèle de confusion contextuelle (bigrammes) différentiable
# ---------------------------------------------------------------------------

class ConfusionModel(nn.Module):
    """
    Embeddings de lettres + table bigramme afin d'estimer un risque contextuel.
    - char_emb: similarité cosinus -> confusion morphologique générale
    - bigram_bias: mémorise spécifiquement les bigrammes à risque (fi, fl, bd, pq, etc.)
    """
    def _init_(self, vocab_size, d=32):
        super()._init_()
        self.emb = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.emb.weight, std=0.05)
        self.bigram_bias = nn.Parameter(torch.zeros(vocab_size, vocab_size))

    def pair_confusion(self, a_ids, b_ids):
        """
        a_ids,b_ids: (B,L) identifiants de lettres voisines
        Retourne C(a,b) \in [0,1]
        """
        ea = self.emb(a_ids)   # (B,L,d)
        eb = self.emb(b_ids)   # (B,L,d)
        sim = F.cosine_similarity(ea, eb, dim=-1)  # (-1,1)
        sim = (sim + 1.0)/2.0  # (0,1)
        bias = torch.sigmoid(self.bigram_bias[a_ids, b_ids])
        return torch.clamp(0.5*sim + 0.5*bias, 0.0, 1.0)

    def context_risk(self, char_ids):
        """
        char_ids: (B,L)
        Retourne R_i (B,L) basé sur voisins gauche/droite et bigrammes.
        """
        B, L = char_ids.shape
        pad = torch.full((B,1), 0, dtype=char_ids.dtype, device=char_ids.device)
        left  = torch.cat([pad, char_ids[:, :-1]], dim=1)
        right = torch.cat([char_ids[:, 1:], pad], dim=1)

        C_left  = self.pair_confusion(char_ids, left)
        C_right = self.pair_confusion(char_ids, right)

        # Bigram context (i-1,i) and (i,i+1)
        C_bi_l = self.pair_confusion(left, char_ids)
        C_bi_r = self.pair_confusion(char_ids, right)

        R = 0.35*C_left + 0.35*C_right + 0.15*C_bi_l + 0.15*C_bi_r
        return torch.clamp(R, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Encodeur de contexte (lettres + regard)
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    def _init_(self, vocab_size, d_char=48, d_gaze=16, d_model=128, n_layers=2, n_heads=4):
        super()._init_()
        self.emb = nn.Embedding(vocab_size, d_char)
        self.proj_gaze = nn.Sequential(
            nn.Linear(6, d_gaze), nn.SiLU(),
            nn.Linear(d_gaze, d_gaze)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_char + d_gaze, nhead=n_heads, batch_first=True,
            dim_feedforward=4*(d_char+d_gaze), activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_char + d_gaze, d_model)

    def forward(self, char_ids, gaze_feats):
        # (B,L) ; (B,L,6)
        x = torch.cat([self.emb(char_ids), self.proj_gaze(gaze_feats)], dim=-1)
        h = self.encoder(x)                  # (B,L,d_char+d_gaze)
        h = torch.relu(self.proj_out(h))     # (B,L,d_model)
        return h

# ---------------------------------------------------------------------------
# Générateur de champ RBF (fixe centres/sigmas) + Warper
# ---------------------------------------------------------------------------

class RBFField(nn.Module):
    """
    Convertit des poids (dx,dy) sur M noyaux RBF en un champ de déplacement (N,H,W,2).
    - centres: (M,2) dans [-1,1]^2 (fixes)
    - sigma: largeur des noyaux
    """
    def _init_(self, M=64, sigma=0.35):
        super()._init_()
        # Centres fixés sur une grille régulière
        s = int(math.sqrt(M))
        xs = np.linspace(-0.85, 0.85, s)
        ys = np.linspace(-0.85, 0.85, s)
        cx, cy = np.meshgrid(xs, ys, indexing="xy")
        centers = np.stack([cx.reshape(-1), cy.reshape(-1)], axis=-1)[:M]
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))  # (M,2)
        self.sigma = sigma

    def forward(self, weights, H, W):
        """
        weights: (N, 2M) -> (dx_weights, dy_weights)
        Retourne flow: (N, H, W, 2) dans l'espace normalisé [-1,1] (pour grid_sample).
        """
        device = weights.device
        N = weights.shape[0]
        M2 = weights.shape[1]
        M = M2 // 2
        wx, wy = weights[:, :M], weights[:, M:]    # (N,M)

        # Grille (H,W,2)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij"
        )
        grid = torch.stack([xs, ys], dim=-1)  # (H,W,2)
        grid = grid.view(1, H, W, 2)          # (1,H,W,2)

        # Distances aux centres (1,H,W,M)
        # g[...,k] = ||(x,y) - c_k||^2
        c = self.centers.to(device).view(1, 1, 1, -1, 2)
        g = (grid.unsqueeze(3) - c).pow(2).sum(-1)  # (1,H,W,M)
        k = torch.exp(-g / (2*self.sigma*self.sigma))  # (1,H,W,M)

        # Combinaison linéaire par wx/wy
        k = k.expand(N, H, W, -1)  # (N,H,W,M)
        dx = (k * wx.view(N, 1, 1, M)).sum(-1, keepdim=True)
        dy = (k * wy.view(N, 1, 1, M)).sum(-1, keepdim=True)
        flow = torch.cat([dx, dy], dim=-1)  # (N,H,W,2)

        # Clamping doux pour rester raisonnable
        flow = torch.clamp(flow, -0.35, 0.35)
        return flow

class Warper(nn.Module):
    def _init_(self):
        super()._init_()

    def forward(self, glyphs, flow):
        """
        glyphs: (N,1,H,W)   ; flow: (N,H,W,2) en coord. normalisées (delta)
        Retourne warped: (N,1,H,W)
        """
        N, _, H, W = glyphs.shape
        device = glyphs.device
        base = make_base_grid(H, W, device).expand(N, -1, -1, -1)  # (N,H,W,2)
        grid = base + flow
        warped = F.grid_sample(glyphs, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

# ---------------------------------------------------------------------------
# Décideur de morphisme (p_change + gating + poids RBF + score confort)
# ---------------------------------------------------------------------------

class MorphDecider(nn.Module):
    def _init_(self, d_model=128, K=4, M=64):
        super()._init_()
        self.K, self.M = K, M
        self.p_change = nn.Linear(d_model, 1)
        self.gate = nn.Linear(d_model, K)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 192), nn.SiLU(),
                nn.Linear(192, 2*M)
            ) for _ in range(K)
        ])
        self.scorer = nn.Sequential(nn.Linear(d_model,128), nn.SiLU(), nn.Linear(128,1))

        # Estimation du gain de séparation attendu (pour le score A_i)
        self.sep_pred = nn.Sequential(
            nn.Linear(3*d_model, 256), nn.SiLU(),
            nn.Linear(256, 1)
        )

    def forward(self, h, h_left, h_right, tau=0.5, hard=False):
        """
        h: (B,L,d)   h_left/h_right: (B,L,d) (décalés)
        Retourne:
          pchg: (B,L)
          gate: (B,L,K)
          weights: (B,L,2M)
          comfort: (B,L)
          exp_sep: (B,L)  (gain de séparation attendu)
        """
        B,L,_ = h.shape
        pchg = torch.sigmoid(self.p_change(h)).squeeze(-1)     # (B,L)

        logits = self.gate(h)                                  # (B,L,K)
        gate = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # (B,L,K)

        W = torch.stack([e(h) for e in self.experts], dim=2)   # (B,L,K,2M)
        weights = torch.einsum('blk,blkm->blm', gate, W)       # (B,L,2M)

        comfort = self.scorer(h).squeeze(-1)                   # (B,L)

        # séparation attendue à partir du contexte (i-1, i, i+1)
        cat_ctx = torch.cat([h_left, h, h_right], dim=-1)
        exp_sep = self.sep_pred(cat_ctx).squeeze(-1)           # (B,L)
        return pchg, gate, weights, comfort, exp_sep

# ---------------------------------------------------------------------------
# Modèle global : encodeur + maxwell + confusion + décideur + warp
# ---------------------------------------------------------------------------

class AdaptiveMorphismModel(nn.Module):
    def _init_(self, vocab_size, d_model=128, K=4, M=64):
        super()._init_()
        self.reliab = MaxwellReliability()
        self.confuse = ConfusionModel(vocab_size)
        self.encoder = ContextEncoder(vocab_size, d_model=d_model)
        self.decider = MorphDecider(d_model=d_model, K=K, M=M)
        self.field = RBFField(M=M, sigma=0.35)
        self.warper = Warper()

        # Coeffs pour score A_i
        self.alpha = 0.9   # poids risque contexte
        self.eta   = 0.6   # poids gain séparation attendu
        self.mu    = 0.03  # coût morphique
        self.tau0  = 0.35  # seuil de présélection
        self.tau   = 0.45  # seuil final

    def _shift(self, x, direction):
        # Décale (B,L,d) vers gauche/droite avec padding zéro
        B,L,D = x.shape
        z = torch.zeros(B,1,D, device=x.device, dtype=x.dtype)
        if direction == 'left':
            return torch.cat([z, x[:, :-1]], dim=1)
        else:
            return torch.cat([x[:, 1:], z], dim=1)

    def forward(self, char_ids, gaze_feats, glyphs, tau=0.5, hard_gate=False,
                return_flows=False):
        """
        char_ids: (B,L) indices de lettres (0 doit être le PAD/blank)
        gaze_feats: (B,L,6) stats regard
        glyphs: (B,L,1,H,W) glyphes à déformer

        Retourne dict:
          out_text: glyphes éventuellement déformés (B,L,1,H,W)
          masks: positions modifiées (B,L)
          logs: dict de scalaires/tensors utiles
          (et si return_flows=True: flow par lettre sélectionnée)
        """
        B,L,_,H,W = glyphs.shape
        device = glyphs.device

        # 1) Mesures Maxwell: D_i, T_i, r_i
        D, T, r = self.reliab(gaze_feats)        # (B,L)

        # 2) Risque contextuel R_i
        R = self.confuse.context_risk(char_ids)  # (B,L)

        # 3) Encodeur contexte -> h_i
        h = self.encoder(char_ids, gaze_feats)   # (B,L,d)
        h_left  = self._shift(h, 'left')
        h_right = self._shift(h, 'right')

        # 4) Décideur -> prob change, gating, poids RBF, score confort, sep attendue
        pchg, gate, weights, comfort, exp_sep = self.decider(h, h_left, h_right, tau=tau, hard=hard_gate)

        # 5) Coût morphique (norme des poids)
        morph_cost = (weights.pow(2).mean(dim=-1))  # (B,L)

        # 6) Score d'activation A_i
        #    présélection (D + alpha R) > tau0
        prelim = (D + self.alpha*R)
        candidates = prelim > self.tau0

        #    score final A = D + aR + eta*exp_sep - mu*cost
        A = D + self.alpha*R + self.eta*torch.tanh(exp_sep) - self.mu*morph_cost
        masks = (A > self.tau) & (r > 0.35) & candidates & (pchg > 0.5)  # (B,L)

        # 7) Warp des positions sélectionnées
        out = glyphs.clone()
        flows_ret = None
        if masks.any():
            idx_b, idx_l = masks.nonzero(as_tuple=True)
            W_sel = weights[idx_b, idx_l]                        # (Nsel,2M)
            glyph_sel = glyphs[idx_b, idx_l]                     # (Nsel,1,H,W)
            flow = self.field(W_sel, H, W)                       # (Nsel,H,W,2)
            warped = self.warper(glyph_sel, flow)                # (Nsel,1,H,W)
            out[idx_b, idx_l] = warped
            if return_flows:
                flows_ret = (idx_b, idx_l, flow)

        logs = {
            "D": D, "T": T, "r": r, "R": R, "pchg": pchg,
            "comfort": comfort, "exp_sep": exp_sep,
            "morph_cost": morph_cost, "A": A, "candidates": candidates
        }
        return out, masks, logs, flows_ret

# ---------------------------------------------------------------------------
# Pertes d'entraînement (préférences, regard, régularisations)
# ---------------------------------------------------------------------------

class LossBundle(nn.Module):
    """
    - L_pref: Bradley-Terry sur paires (variante A vs B) avec score de confort
    - L_gaze: réduction des durées/dispersion après morphisme (proxy régression)
    - L_tv: total-variation du champ (via flux renvoyé)
    - L_maxwell: prior qui pénalise la norme du champ proportionnel à T
    """
    def _init_(self, w_pref=1.0, w_gaze=0.7, w_tv=0.05, w_max=0.05):
        super()._init_()
        self.w_pref = w_pref
        self.w_gaze = w_gaze
        self.w_tv = w_tv
        self.w_max = w_max
        self.mse = nn.MSELoss(reduction='mean')

    def pref_loss(self, score_A, score_B, y):
        """
        score_A/score_B: (N,) logits de confort des deux variantes
        y in {0,1}: 1 si A préféré, 0 si B préféré
        """
        return F.binary_cross_entropy_with_logits(score_A - score_B, y.float())

    def gaze_loss(self, dur_before, dur_after, disp_before, disp_after):
        """
        On encourage la baisse de durées et de dispersion (après < avant).
        """
        l1 = self.mse(dur_after, torch.clamp(dur_before - 0.10, min=0.0))
        l2 = self.mse(disp_after, torch.clamp(disp_before - 0.05, min=0.0))
        return l1 + l2

    def forward(self, logs, flows_info=None, T=None,
                pref_pairs=None, gaze_before=None, gaze_after=None):
        """
        logs: dict du modèle (contient comfort, etc.)
        flows_info: (idx_b, idx_l, flow) ou None
        T: (B,L) températures
        pref_pairs: dict optionnel avec:
          - scoreA, scoreB, y  (N,) chacun
        gaze_before/gaze_after: dict optionnel avec:
          - dur, disp   (B,L)
        """
        total = torch.tensor(0.0, device=logs["A"].device)

        # L_pref
        if pref_pairs is not None:
            lp = self.pref_loss(pref_pairs["scoreA"], pref_pairs["scoreB"], pref_pairs["y"])
            total = total + self.w_pref * lp

        # L_gaze
        if (gaze_before is not None) and (gaze_after is not None):
            lg = self.gaze_loss(gaze_before["dur"], gaze_after["dur"],
                                gaze_before["disp"], gaze_after["disp"])
            total = total + self.w_gaze * lg

        # L_tv + L_maxwell sur le flow des positions sélectionnées
        if (flows_info is not None) and (T is not None):
            idx_b, idx_l, flow = flows_info  # flow: (Nsel,H,W,2)
            if flow.numel() > 0:
                ltv = tv_norm(flow)
                total = total + self.w_tv * ltv

                # Maxwell prior: ||flow||^2 pondéré par T local moyen
                flow_energy = (flow.pow(2).sum(dim=-1)).mean(dim=(1,2))  # (Nsel,)
                T_sel = T[idx_b, idx_l]                                  # (Nsel,)
                lmax = (flow_energy * (1.0 + 0.8*T_sel)).mean()
                total = total + self.w_max * lmax

        return total

# ---------------------------------------------------------------------------
# Démo d'inférence sur "fils" (synthétique)
# ---------------------------------------------------------------------------

if _name_ == "_main_":
    torch.manual_seed(42)

    # 1) Vocab & encodage simple
    alphabet = ["<pad>"] + list("abcdefghijklmnopqrstuvwxyz")
    stoi = {c:i for i,c in enumerate(alphabet)}
    itos = {i:c for c,i in stoi.items()}
    vocab_size = len(alphabet)

    # 2) Mot: "fils"
    word = "fils"
    B, L = 1, len(word)
    H, W = 96, 96

    # Ids lettres
    char_ids = torch.tensor([[stoi[c] for c in word]], dtype=torch.long)

    # 3) Glyphes synthétiques: blobs centraux (placeholder pour de vrais glyphes)
    #    Ici: petits carrés pour démontrer le warp. Remplace par de vrais glyphs (0–1).
    glyphs = torch.zeros(B, L, 1, H, W)
    for i in range(L):
        y0, y1 = 30, 66
        x0, x1 = 30 + 4*i, 66 + 4*i  # légère oblique pour voir l'effet
        glyphs[0, i, 0, y0:y1, x0:x1] = 1.0

    # 4) Gaze synthétique : rendre "fi" plus difficile (durée/dispersion élevées)
    gaze_feats = torch.zeros(B, L, 6)
    # features: dur, disp, reg, v_mean, v_var, fix_ratio
    base = torch.tensor([0.30, 0.20, 0.10, 40.0, 80.0, 0.60])  # baseline
    for i, c in enumerate(word):
        gaze_feats[0, i] = base
    # aggraver sur 'f' et 'i'
    gaze_feats[0, 0, 0] = 0.45  # dur f
    gaze_feats[0, 1, 0] = 0.42  # dur i
    gaze_feats[0, 0, 1] = 0.30  # disp f
    gaze_feats[0, 1, 1] = 0.28  # disp i
    gaze_feats[0, 0, 4] = 140.0 # v_var f -> T ↑
    gaze_feats[0, 1, 4] = 130.0 # v_var i -> T ↑

    # 5) Modèle
    model = AdaptiveMorphismModel(vocab_size=vocab_size, d_model=128, K=4, M=49)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    char_ids = char_ids.to(device)
    gaze_feats = gaze_feats.to(device)
    glyphs = glyphs.to(device)

    # 6) Inférence
    out, masks, logs, flows = model(
        char_ids=char_ids,
        gaze_feats=gaze_feats,
        glyphs=glyphs,
        tau=0.5,
        hard_gate=False,
        return_flows=True
    )

    print("Mot:", word)
    print("Positions modifiées (mask):", masks.int().cpu().numpy())  # 1 si déformée
    print("Scores D (difficulté):", logs["D"].detach().cpu().numpy().round(3))
    print("Risque contexte R:", logs["R"].detach().cpu().numpy().round(3))
    print("Fiabilité r:", logs["r"].detach().cpu().numpy().round(3))
    print("Score A:", logs["A"].detach().cpu().numpy().round(3))

    # 7) Exemple de calcul de pertes (optionnel, schéma)
    losses = LossBundle(w_pref=1.0, w_gaze=0.7, w_tv=0.05, w_max=0.05).to(device)

    # Paires de préférences simulées: variante A gagnante (y=1)
    # Ici, on réutilise le score de confort comme proxy de logit
    # (dans un vrai protocole, ces scores proviennent de deux passes A/B).
    scoreA = logs["comfort"].mean(dim=0)  # (L,) proxy
    scoreB = logs["comfort"].mean(dim=0) - 0.2  # B moins confortable
    y = torch.ones(L, device=device)  # A préféré

    # Gaze avant/après (ici on simule un petit gain après)
    dur_before = gaze_feats[..., 0]
    disp_before = gaze_feats[..., 1]
    dur_after = torch.clamp(dur_before - 0.06, min=0.05)
    disp_after = torch.clamp(disp_before - 0.03, min=0.03)

    pref_pairs = {"scoreA": scoreA.flatten(), "scoreB": scoreB.flatten(), "y": y}
    gaze_before = {"dur": dur_before, "disp": disp_before}
    gaze_after = {"dur": dur_after, "disp": disp_after}

    loss = losses(
        logs=logs,
        flows_info=flows,
        T=logs["T"],
        pref_pairs=pref_pairs,
        gaze_before=gaze_before,
        gaze_after=gaze_after
    )
    print("Loss totale (démo):", float(loss.detach().cpu()))