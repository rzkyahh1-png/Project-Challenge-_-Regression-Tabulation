import numpy as np
import pandas as pd
from scipy import stats

# ============================ PENGATURAN ============================
FILE_CSV = "concrete_data.csv"        # nama file dataset dari Google Drive
SEED = 2026                      # CATAT seed ini di laporan (ganti sesukamu)
FRAKSI_SAMPEL = 0.90             # 85%-90% sesuai panduan
OUTPUT_SAMPEL = "sampel_beton.csv"

# Dua variabel untuk tabulasi silang (pilih dua dari tiga variabel numerik)
VAR_CT_1 = "cement"
VAR_CT_2 = "strength"
# ====================================================================

FILE_CSV = "C:/Users/UseR/Downloads/concrete_data.csv"
def cari_kolom(df, kata_kunci):
    """Cari nama kolom asli yang mengandung kata kunci (tidak peka huruf besar/kecil)."""
    for kol in df.columns:
        if kata_kunci.lower() in str(kol).lower():
            return kol
    raise KeyError(f"Kolom dengan kata '{kata_kunci}' tidak ditemukan. Kolom ada: {list(df.columns)}")


# ------------------------- 1. RANDOM SAMPLING -------------------------
def ambil_sampel(df):
    n_total = len(df)
    n_sampel = round(FRAKSI_SAMPEL * n_total)          # 0.90 * 1030 = 927
    sampel = df.sample(n=n_sampel, replace=False, random_state=SEED)
    sampel = sampel.reset_index(drop=True)
    sampel.to_csv(OUTPUT_SAMPEL, index=False)
    print("=" * 70)
    print("1. RANDOM SAMPLING")
    print("=" * 70)
    print(f"Data awal           : {n_total}")
    print(f"Data digunakan      : {n_sampel} ({FRAKSI_SAMPEL:.0%})")
    print(f"Tidak diikutsertakan: {n_total - n_sampel}")
    print(f"Random seed         : {SEED}")
    print(f"Sampel disimpan ke  : {OUTPUT_SAMPEL}\n")
    return sampel


# ------------------------- 2. REGRESI OLS -----------------------------
def regresi_ols(sampel, kol_y, kol_x1, kol_x2):
    y = sampel[kol_y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(sampel)),
                         sampel[kol_x1].to_numpy(dtype=float),
                         sampel[kol_x2].to_numpy(dtype=float)])
    n, k = X.shape                                     # k = jumlah parameter (3)

    # OLS: b = (X'X)^-1 X'y
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y

    y_hat = X @ b
    resid = y - y_hat
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = sst - sse
    r2 = 1 - sse / sst
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)
    mse = sse / (n - k)

    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_hit = b / se
    p_t = 2 * (1 - stats.t.cdf(np.abs(t_hit), df=n - k))

    f_hit = (ssr / (k - 1)) / mse
    p_f = 1 - stats.f.cdf(f_hit, k - 1, n - k)

    nama = ["Konstanta (b0)", f"{kol_x1} (b1)", f"{kol_x2} (b2)"]
    print("=" * 70)
    print("2. REGRESI LINEAR BERGANDA (OLS)")
    print("=" * 70)
    print(f"Y  = {kol_y}")
    print(f"X1 = {kol_x1}")
    print(f"X2 = {kol_x2}")
    print(f"n  = {n}\n")
    tabel = pd.DataFrame({"Koefisien": b, "Std.Error": se, "t-hitung": t_hit, "p-value": p_t},
                         index=nama)
    print(tabel.round(6).to_string())
    print(f"\nR-squared     : {r2:.4f}")
    print(f"Adj R-squared : {r2_adj:.4f}")
    print(f"F-hitung      : {f_hit:.4f}  (p-value = {p_f:.6g})")
    print("\nPersamaan regresi:")
    print(f"Y_hat = {b[0]:.4f} + {b[1]:.4f}*X1 + {b[2]:.4f}*X2")
    print("\nInterpretasi singkat (isi dengan kata-katamu di laporan):")
    print(f"- b0: dugaan Y saat X1 = 0 dan X2 = 0 = {b[0]:.4f}")
    print(f"- b1: tiap X1 naik 1 satuan (X2 tetap), Y berubah {b[1]:.4f}")
    print(f"- b2: tiap X2 naik 1 satuan (X1 tetap), Y berubah {b[2]:.4f}\n")



# ------------------------- 3. CROSS TABULATION ------------------------
def kategorikan_tertile(seri, nama):
    """Rendah <= P33.33 ; Sedang <= P66.67 ; Tinggi > P66.67"""
    batas1, batas2 = np.percentile(seri, [100 / 3, 200 / 3])
    print(f"Batas kategori {nama}: P33.33 = {batas1:.4f} ; P66.67 = {batas2:.4f}")
    if batas1 == batas2:
        print(f"  PERINGATAN: batas {nama} kembar (banyak nilai sama). "
              "Kategori 'sedang' akan kosong. Pilih variabel lain atau jelaskan di laporan.")
    kat = pd.cut(seri, bins=[-np.inf, batas1, batas2, np.inf],
                 labels=["Rendah", "Sedang", "Tinggi"])
    return kat


def cross_tab(sampel, kol_a, kol_b):
    print("=" * 70)
    print("3. CROSS TABULATION 3 x 3 (TERTILE)")
    print("=" * 70)
    kat_a = kategorikan_tertile(sampel[kol_a], kol_a)
    kat_b = kategorikan_tertile(sampel[kol_b], kol_b)

    obs = pd.crosstab(kat_a, kat_b, dropna=False)
    obs = obs.reindex(index=["Rendah", "Sedang", "Tinggi"],
                      columns=["Rendah", "Sedang", "Tinggi"], fill_value=0)
    obs.index.name, obs.columns.name = kol_a, kol_b

    n = obs.to_numpy().sum()
    tot_baris = obs.sum(axis=1).to_numpy()
    tot_kolom = obs.sum(axis=0).to_numpy()
    harapan = np.outer(tot_baris, tot_kolom) / n          # E = (total baris x total kolom) / n
    exp = pd.DataFrame(harapan, index=obs.index, columns=obs.columns)

    print("\nFrekuensi observasi (O):")
    print(obs.assign(Total=tot_baris).to_string())
    print("\nFrekuensi harapan (E):")
    print(exp.round(3).to_string())

    # Chi-square = sum (O-E)^2 / E
    with np.errstate(divide="ignore", invalid="ignore"):
        kontribusi = np.where(harapan > 0, (obs.to_numpy() - harapan) ** 2 / harapan, 0.0)
    chi2 = float(kontribusi.sum())
    df_ = (3 - 1) * (3 - 1)
    p_val = 1 - stats.chi2.cdf(chi2, df_)

    # Koefisien kontingensi C = sqrt(chi2 / (chi2 + n))
    C = np.sqrt(chi2 / (chi2 + n))
    C_maks = np.sqrt((3 - 1) / 3)                         # batas maksimum untuk tabel 3x3

    print(f"\nn                      : {n}")
    print(f"Chi-Square             : {chi2:.4f}")
    print(f"db                     : {df_}")
    print(f"p-value                : {p_val:.6g}")
    print(f"Koefisien kontingensi C: {C:.4f}")
    print(f"C maksimum (3x3)       : {C_maks:.4f}")
    print(f"C / C maks             : {C / C_maks:.4f}")
    if (harapan < 5).any():
        print("Catatan: ada sel dengan frekuensi harapan < 5 (asumsi Chi-Square kurang terpenuhi).")
    print()


# ------------------------------- MAIN ---------------------------------
def main():
    df = pd.read_csv(FILE_CSV)
    kol_y = cari_kolom(df, "strength")
    kol_x1 = cari_kolom(df, "cement")
    kol_x2 = cari_kolom(df, "age")
    df = df[[kol_x1, kol_x2, kol_y]].dropna()

    sampel = ambil_sampel(df)                  # satu sampel untuk semua analisis
    regresi_ols(sampel, kol_y, kol_x1, kol_x2)

    kol_a = cari_kolom(sampel, VAR_CT_1)
    kol_b = cari_kolom(sampel, VAR_CT_2)
    cross_tab(sampel, kol_a, kol_b)


if __name__ == "__main__":
    main()