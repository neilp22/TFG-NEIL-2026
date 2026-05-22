import numpy as np
from scipy.stats import wilcoxon, ttest_1samp, ttest_rel


def test_vs_random(aucs: list, model_name: str, alpha: float = 0.05) -> dict:
    """
    H0: AUC = 0.5 (el modelo no supera el azar).
    Usa t-test de una muestra sobre los AUC de los splits.
    Con n=4 splits el poder estadístico es bajo (~0.20 para delta=0.05),
    por lo que p>0.05 es el resultado esperado incluso si el modelo añade algo.
    """
    arr = np.array(aucs)
    t_stat, p_value = ttest_1samp(arr, popmean=0.5)
    mean_auc = arr.mean()
    std_auc  = arr.std(ddof=1)
    se       = std_auc / np.sqrt(len(arr))
    ci_low   = mean_auc - 1.96 * se
    ci_high  = mean_auc + 1.96 * se

    return {
        'model':       model_name,
        'n_splits':    len(arr),
        'auc_mean':    round(mean_auc, 4),
        'auc_std':     round(std_auc, 4),
        'ci_95':       (round(ci_low, 4), round(ci_high, 4)),
        't_stat':      round(t_stat, 4),
        'p_value':     round(p_value, 4),
        'significant': p_value < alpha,
        'conclusion':  'SUPERA el azar' if (p_value < alpha and mean_auc > 0.5)
                       else 'NO supera el azar estadísticamente',
    }


def compare_models(aucs_a: list, aucs_b: list,
                   name_a: str, name_b: str,
                   alpha: float = 0.05) -> dict:
    """
    Compara dos modelos con test de Wilcoxon pareado por splits.
    Con n<5 pares Wilcoxon no es válido; se cae a t-test pareado.
    """
    arr_a = np.array(aucs_a)
    arr_b = np.array(aucs_b)

    if len(arr_a) < 5:
        stat, p = ttest_rel(arr_a, arr_b)
        test_used = 't-test pareado'
    else:
        stat, p = wilcoxon(arr_a, arr_b)
        test_used = 'Wilcoxon'

    return {
        'comparison':  f'{name_a} vs {name_b}',
        'test_used':   test_used,
        'diff_mean':   round(float(arr_a.mean() - arr_b.mean()), 4),
        'stat':        round(float(stat), 4),
        'p_value':     round(float(p), 4),
        'significant': p < alpha,
        'conclusion':  (f'{name_a} es SIGNIFICATIVAMENTE mejor'
                        if p < alpha and arr_a.mean() > arr_b.mean()
                        else f'Diferencia NO significativa (p={p:.3f})'),
    }


def print_full_report(models_aucs: dict, alpha: float = 0.05):
    """
    Imprime tabla de tests H0: AUC=0.5 para cada modelo.
    models_aucs = {'XGB+Sent': [0.43, 0.55, ...], 'LightGBM': [...], ...}
    """
    print("\n" + "=" * 72)
    print("TESTS ESTADÍSTICOS — COMPARACIÓN VS AZAR (H0: AUC = 0.5)")
    print(f"Nivel de significancia: α = {alpha}")
    print("=" * 72)
    print(f"{'Modelo':<22} {'AUC medio':>10} {'IC 95%':>22} {'p-value':>10} {'Sig?':>6}")
    print("-" * 72)

    results = []
    for name, aucs in models_aucs.items():
        r = test_vs_random(aucs, name, alpha)
        results.append(r)
        sig = "SI" if r['significant'] else "NO"
        ci  = f"[{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}]"
        print(f"{name:<22} {r['auc_mean']:>10.4f} {ci:>22} {r['p_value']:>10.4f} {sig:>6}")

    print()
    n_sig = sum(r['significant'] for r in results)
    if n_sig == 0:
        print("CONCLUSIÓN: Ningún modelo supera el azar estadísticamente.")
        print("Resultado consistente con la Hipótesis de Mercados Eficientes (EMH).")
        print("Nota: con n=4 splits el poder estadístico es ~0.20 para AUC+0.05,")
        print("por lo que la ausencia de significancia no descarta mejora real.")
    else:
        print(f"CONCLUSIÓN: {n_sig}/{len(results)} modelos superan el azar (α={alpha}).")
    print("=" * 72)

    return results


def power_analysis(effect_size: float = 0.05, n_splits: int = 4,
                   alpha: float = 0.05) -> float:
    """
    Estima el poder estadístico del t-test de una muestra.
    Asume AUC ~ N(0.5 + effect_size, sigma) con sigma estimada de los datos.
    Devuelve la probabilidad de detectar una mejora real de `effect_size`.

    Con n=4 y efecto realista (delta=0.05, sigma=0.07):
      poder ≈ 0.18 — la prueba es casi ciega a mejoras pequeñas.
    """
    from scipy.stats import t as t_dist, nct

    sigma_typical = 0.07   # varianza típica observada en splits BTC
    ncp   = effect_size / (sigma_typical / np.sqrt(n_splits))
    t_crit = t_dist.ppf(1 - alpha / 2, df=n_splits - 1)
    power = 1 - nct.cdf(t_crit, df=n_splits - 1, nc=ncp)
    return round(power, 4)


if __name__ == '__main__':
    # Ejemplo de uso con AUCs hipotéticos similares a los resultados reales
    example = {
        'XGB+Sent':    [0.43, 0.55, 0.51, 0.58],
        'XGB Precio':  [0.48, 0.53, 0.49, 0.54],
        'LightGBM':    [0.47, 0.52, 0.50, 0.51],
        'LSTM':        [0.45, 0.54, 0.52, 0.50],
        'ARIMA':       [0.44, 0.51, 0.48, 0.52],
    }

    print_full_report(example)

    print("\n" + "=" * 72)
    print("COMPARACIONES PAREADAS ENTRE MODELOS")
    print("=" * 72)
    pairs = [
        ('XGB+Sent', 'XGB Precio'),
        ('XGB+Sent', 'LightGBM'),
        ('XGB+Sent', 'ARIMA'),
    ]
    for a, b in pairs:
        r = compare_models(example[a], example[b], a, b)
        print(f"\n{r['comparison']} ({r['test_used']})")
        print(f"  Diferencia media: {r['diff_mean']:+.4f}")
        print(f"  p-value: {r['p_value']:.4f}  →  {r['conclusion']}")

    print()
    pow_ = power_analysis(effect_size=0.05, n_splits=4)
    print(f"Poder estadístico (delta=0.05, n=4 splits): {pow_:.2%}")
    print("Con este poder, el 80% de las mejoras reales de AUC+0.05 NO se detectarían.")
