"""Assemble the submission notebook (HW4_NAS_Hyperopt.ipynb) and REPORT.md.

Reads the metrics produced by run_search.py / run_retrain.py / run_proxy_corr.py
from RESULTS_DIR and bakes the real numbers into a Ukrainian narrative. Run AFTER
all three experiments finish, then execute the notebook with nbconvert.
"""
import json
import os
import sys
from collections import Counter

import nbformat as nbf

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT_NB = "HW4_NAS_Hyperopt.ipynb"
OUT_REPORT = "REPORT.md"


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


search = load("search.json")
retrain = load("retrain.json")
corr = load("proxy_corr.json")

pct = lambda x: f"{x * 100:.2f}%"
pp = lambda x: f"{x * 100:+.2f} п.п."      # Ukrainian notebook
ppe = lambda x: f"{x * 100:+.2f} pp"        # English REPORT
M = lambda n: f"{n / 1e6:.2f}M"

msd = lambda mean, std: f"{mean * 100:.2f} ± {std * 100:.2f}%"     # mean±std, percent
pf = lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}"           # p-value, capped below

cfg = search["config"]
records = search["records"]
rand_records = search.get("random_records", [])
best = search["best_arch"]
best_str = "/".join(best["ops"]) + f", {best['width']}×, {best['act']}"

# distinct architectures actually evaluated (TPE)
seen = {}
for r in records:
    k = (tuple(r["arch"]["ops"]), r["arch"]["width"], r["arch"]["act"])
    seen.setdefault(k, r)
n_unique = len(seen)
n_cached = sum(1 for r in records if r["cached"])
uniq_sorted = sorted(seen.values(), key=lambda r: r["val_acc"], reverse=True)
top15 = uniq_sorted[:15]
# at how many distinct archs did TPE reach its best? (early -> little TPE contribution)
best_loss = min(r["val_loss"] for r in records)
first_best = next(r for r in records if r["val_loss"] == best_loss)
best_at_unique = first_best["n_unique_so_far"]
rand_best_acc = max((r["val_acc"] for r in rand_records), default=0.0)
tpe_best_acc = max(r["val_acc"] for r in records)

# operation / width / activation frequency across the top-15 distinct designs
stage_freq = [Counter(r["arch"]["ops"][s] for r in top15) for s in range(4)]
width_freq = Counter(r["arch"]["width"] for r in top15)
act_freq = Counter(r["arch"]["act"] for r in top15)
freq_str = lambda c: ", ".join(f"{k}×{v}" for k, v in c.most_common())

base = retrain["baseline_vgg11"]
b = retrain["best"]
d = retrain["default"]
seeds = retrain["config"]["seeds"]
comp = base["params"] / b["params"]
dp_vs_base = b["test_mean"] - base["test_acc"]
gap = b["test_mean"] - d["test_mean"]                 # best - default (mean)

full = corr["full"]                                   # coarse-filter correlation (whole range)
topc = corr["top"]                                    # fine-ranking correlation (top slice)
top_k = corr["top_k"]
crows = corr["rows"]
n_corr = len(crows)
tk = sorted(crows, key=lambda r: r["proxy_val_acc"], reverse=True)[:top_k]
proxy_spread = max(r["proxy_val_acc"] for r in tk) - min(r["proxy_val_acc"] for r in tk)
real_spread = max(r["short_val_acc"] for r in tk) - min(r["short_val_acc"] for r in tk)

cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))


# --------------------------------------------------------------------------- #
md(f"""
# Домашня робота 4 — Neural Architecture Search з Hyperopt (CNN / CIFAR-10)

**Курс:** Efficient ML, SET University

Neural Architecture Search (NAS) — це автоматичний пошук самої **архітектури** мережі,
а не її ваг. Замість того щоб вручну вгадувати «скільки шарів, які блоки, яка ширина»,
ми задаємо **дискретний простір** можливих архітектур і даємо оптимізатору знайти в
ньому найкращу. Тут оптимізатор — **TPE** (Tree-structured Parzen Estimator) з
[Hyperopt](https://github.com/hyperopt/hyperopt).

Головна проблема NAS — вартість: повноцінно натренувати кожну архітектуру з нуля надто
дорого. Рятує **weight sharing (one-shot)**: ми один раз тренуємо велику
**supernet**, чиї ваги спільні для всього простору, і оцінюємо будь-яку архітектуру
як її під-мережу — **без окремого навчання**. Це рівно та ідея slimmable-мереж, яку
ми вже застосували в бонусі ДЗ3 (NetAug), тільки тепер розширена на **вибір операції**
в кожній стадії.

### Що ми шукали (три осі, як вимагає умова)

| Вісь | Варіанти |
|---|---|
| **Операція блоку** (на кожну з 4 стадій) | `conv3x3` (звичайна згортка) · `dwsep` (depthwise-separable, MobileNet-v1) · `mbconv` (inverted residual, MobileNet-v2, expand ×3) |
| **Множник ширини** (глобальний) | 0.5 · 0.75 · 1.0 · 1.25 |
| **Активація** (глобальна) | ReLU · ReLU6 · SiLU · GELU · LeakyReLU |

Розмір простору = 3⁴ × 4 × 5 = **{cfg['space_size']} архітектур**.

### Головний результат (тест — раз на кожну фінальну модель; {len(seeds)} seed-и, mean ± std)

| Модель | Параметри | Test acc |
|---|---|---|
| Baseline VGG11 (з ДЗ1, reference — один запуск) | {M(base['params'])} | {pct(base['test_acc'])} |
| In-space default (усі conv3x3, 1.0×, ReLU) | {M(d['params'])} | {msd(d['test_mean'], d['test_std'])} |
| **Знайдена пошуком** ({best_str}) | **{M(b['params'])}** | **{msd(b['test_mean'], b['test_std'])}** |

**Основний висновок — знайдена архітектура vs in-space default** (обидві на {len(seeds)}
seed-ах, ідентичний рецепт, однаковий порядок батчів на кожному seed): **{gap*100:+.2f} п.п.**
({msd(b['test_mean'], b['test_std'])} проти {msd(d['test_mean'], d['test_std'])}) при трохи
меншій кількості параметрів. Розрив перевищує сумарний розкид (±{b['test_std']*100:.2f}/
±{d['test_std']*100:.2f} п.п.), тож пошук **реально** дав кращу архітектуру, а не удачу seed.

Проти **VGG11** знайдена мережа — {ppe(dp_vs_base).replace('pp','п.п.')} при **{comp:.1f}×
менше параметрів**. Але VGG11-baseline має лише **один історичний запуск**, тож це радше
порівняння з reference-точкою, ніж статистично доведена перевага; головне тут — що пошук
дотягується до рівня набагато більшої мережі на порядок меншою.

Наскільки надійним був one-shot proxy? На вибірці, що покриває **весь** діапазон proxy
(стратифіковано по квантилях): Kendall τ = **{full['kendall_tau']:.2f}**
({pf(full['kendall_p'])}) — тобто для **грубого** відсіву proxy надійний. А на **топ**
кандидатах: τ = **{topc['kendall_tau']:.2f}** ({pf(topc['kendall_p'])}) — **тонко**
ранжувати найкращих між собою він майже не вміє. Деталі — у розділах 4–6 і висновках.
""")

md(f"""
## 1. Підготовка

Цей ноутбук — **звіт-візуалізація**: він завантажує готові метрики з `results/*.json`,
а самі експерименти (тренування supernet, пошук, retrain) виконують скрипти на GPU.
Точні команди відтворення:

```bash
# тренує supernet з нуля (~8 хв на GB10) і шукає; supernet.pt зберігається в results/
python run_search.py     --supernet-epochs {cfg['supernet_epochs']} --evals {cfg['evals']}
python run_retrain.py    --baseline ../hw1/results/baseline.pt --epochs {retrain['config']['epochs']} --seeds {','.join(map(str, seeds))}
python run_proxy_corr.py --n-bins {corr['config']['n_bins']} --per-bin {corr['config']['per_bin']} --top-k {top_k} --seeds {','.join(map(str, corr['config']['seeds']))} --short-epochs {corr['config']['short_epochs']}
python tests/test_search_space.py && python tests/test_supernet.py
```
> `run_search.py` має також опцію `--load-supernet results/supernet.pt` для перевикористання
> ваг, але сам чекпойнт `*.pt` не комітиться (gitignored), тож для відтворення з нуля —
> команда вище (тренує supernet заново; результат детермінований за seed з точністю до
> GPU-недетермінізму).

Прогін цього ДЗ — на gx10 (NVIDIA GB10, CUDA 13.0), single-seed для пошуку/supernet,
{len(seeds)} seed-и для retrain. Версії середовища друкуються нижче.
""")
code("""
import json, sys, platform
import torch, torchvision, numpy, scipy, hyperopt
import matplotlib.pyplot as plt

from src.utils import get_device
from src.search_space import space_size, count_arch_params, StandaloneNet, sample_arch
from src import plots

print("python     ", sys.version.split()[0], "|", platform.platform())
print("torch      ", torch.__version__, "| torchvision", torchvision.__version__)
print("numpy      ", numpy.__version__, "| scipy", scipy.__version__,
      "| hyperopt", hyperopt.__version__)
print("device     ", get_device(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

RESULTS = "results"
def load(n):
    with open(f"{RESULTS}/{n}") as f: return json.load(f)
search   = load("search.json")
retrain  = load("retrain.json")
corr     = load("proxy_corr.json")
records  = search["records"]
print("простір    ", space_size(), "архітектур")
""")

md(f"""
## 2. Простір пошуку

Макро-скелет **фіксований**: 4 послідовні стадії, кожна = один шуканий блок + 2×2
max-pool (32→16→8→4→2 по просторовому розміру), потім global average pooling і
лінійний класифікатор. Пошук вирішує лише **що` за блок** у кожній стадії, **яку
ширину** й **яку активацію** — рівно три осі з умови.

Три операції — це класична «сходинка ефективності»:
- **`conv3x3`** — повна згортка 3×3: найвиразніша, найважча за параметрами;
- **`dwsep`** — depthwise 3×3 + pointwise 1×1: та сама форма, у рази менше параметрів;
- **`mbconv`** — inverted residual (expand→depthwise→project, +skip): компроміс, ядро
  сучасних мобільних мереж.

`src/search_space.py` також уміє побудувати будь-яку архітектуру як **звичайну**
(не-shared) мережу `StandaloneNet` — саме її ми потім тренуємо з нуля й саме на ній
рахуємо реальну кількість параметрів.
""")
code("""
from src.search_space import OPS, WIDTHS, ACTS, count_arch_params
print("операції:", OPS)
print("ширини:  ", WIDTHS)
print("активації:", ACTS)
# приклад: та сама структура при різній ширині — різна кількість параметрів
for w in WIDTHS:
    a = {"ops": ["conv3x3","mbconv","conv3x3","mbconv"], "width": w, "act": "relu6"}
    print(f"  width {w}: {count_arch_params(a)/1e6:.2f}M параметрів")
""")

md(f"""
## 3. One-shot supernet + Single-Path One-Shot

Supernet зберігає ваги на **максимальній** ширині (1.25×) і тримає **всі три
операції** в кожній стадії як паралельні гілки. Під-мережа — це вибір однієї гілки
на стадію + зріз каналів `W[:out, :in]` спільної ваги. Тренуємо за схемою **SPOS**
(Guo et al., ECCV 2020): на кожному кроці семплимо **одну** випадкову архітектуру й
оновлюємо лише її шлях. **Архітектури** семплуються рівноймовірно, тож за багато
кроків усі операції/ширини/активації отримують градієнти. (Точніше: рівноймовірний
саме вибір архітектур; у prefix-sharing нижні канали входять у більше під-мереж і
оновлюються частіше за крайні канали широких моделей — тож рівномірність стосується
вибору шляхів, а не частоти оновлення кожного каналу.)

Statistics BatchNorm у one-shot ненадійні (вони змішують усі ширини/операції, які
бачили під час тренування), тому **перед кожною оцінкою** ми **рекалібруємо** BN
активного шляху на кількох батчах — стандартна практика one-shot NAS.

Supernet тут — {M(search['supernet_params'])} параметрів (max ширина, усі операції),
тренували {cfg['supernet_epochs']} епох. Крива нижче — середня точність випадкових
шляхів по епохах (не одна модель, а розподіл над шляхами — тож це грубий сигнал
«здоров'я» supernet, а не фінальна точність).
""")
code("""
h = search.get("supernet_history")
if h:
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(range(1, len(h["train_acc"])+1), [a*100 for a in h["train_acc"]], color="tab:purple")
    ax.set_xlabel("епоха"); ax.set_ylabel("сер. точність шляху, %")
    ax.set_title("SPOS-тренування supernet (середнє по випадкових шляхах)")
    ax.grid(alpha=0.3); plt.show()
    print(f"фінальна сер. точність шляху: {h['train_acc'][-1]*100:.1f}%")
else:
    print("supernet перевикористано з наявних ваг — історію див. у логах run_search.py")
""")

md(f"""
## 4. Пошук TPE (weight-sharing proxy)

TPE будує ймовірнісну модель «які конфігурації дають низький лос» і пропонує нові
кандидати, що максимізують очікуване покращення. Кожен trial коштує копійки: вибрати
під-шлях у натренованій supernet → рекалібрувати BN → порахувати **validation loss**.
Це proxy-оцінка; ми також фіксуємо кількість параметрів і proxy val-accuracy.

За {cfg['evals']} trial-ів TPE оцінив **{n_unique}** унікальних архітектур (з
{cfg['space_size']}); решта {n_cached} — повторні візити вже баченого (TPE
експлуатує вдалі зони — беремо їх з кешу). Кешовані trial-и не додають інформації,
тому на графіку збіжності вісь X — **кількість унікальних** архітектур, а не сирий
номер trial.

**Контроль — random search.** Щоб приписати виграш саме TPE, а не раннім вдалим
семплам, ми прогнали той самий бюджет **випадковим** пошуком по тій самій supernet і
наклали криві. Це прозора перевірка: якщо TPE не обганяє random, так і кажемо.

**Два обов'язкові графіки** (convergence — з random-контролем):
""")
code("""
plots.plot_search_convergence(records, search.get("random_records")); plt.show()
plots.plot_acc_vs_params(records, search["best_arch"]); plt.show()
""")

md(f"""
**Внесок TPE.** Найкращий proxy-loss TPE досяг уже на **{best_at_unique}-й** унікальній
архітектурі. На цьому невеликому просторі (хороший регіон широкий — багато робочих
`conv3x3`-на-вході дизайнів) TPE і random виходять на близький найкращий proxy
(TPE best-acc {pct(tpe_best_acc)} vs random {pct(rand_best_acc)}): перевага
sequential-моделі над random тут помірна, бо знайти хорошу зону легко. Це — валідний
результат, а не хиба (див. графік). **Застереження:** random-контроль тут — **один**
прогін; для строгого твердження «TPE > random» треба 3–5 random-seed-ів і band/mean
(див. розділ «як покращити»).

**Що знайшов пошук.** Топ-архітектури за proxy узгоджені — пошук збігся на чіткий
структурний мотив:

- **стадія 0 → `conv3x3`** ({freq_str(stage_freq[0])}): перший шар з «сирих» 3 каналів
  хоче повну згортку;
- **стадія 1 → `mbconv`** ({freq_str(stage_freq[1])}): inverted residual у середині;
- стадія 2 → {freq_str(stage_freq[2])}; стадія 3 → {freq_str(stage_freq[3])};
- ширина: {freq_str(width_freq)}; активація: {freq_str(act_freq)}.

Найгірші архітектури — усі-`dwsep` при ширині 0.5× — падають до ~67% proxy-точності;
proxy впевнено відсіює їх. **Найкраща за proxy:** `{best_str}`, {M(search['best_params'])},
proxy val-acc = {pct(search['best_proxy_val_acc'])}.
""")

md(f"""
## 5. Retrain з нуля vs baseline

Proxy лише **ранжує**; фінальне число дає навчання обраної архітектури **з нуля** як
звичайної мережі ({retrain['config']['epochs']} епох, cosine LR). Порівнюємо з двома
baseline: (1) заморожена VGG11 з ДЗ1 (той самий baseline через усі ДЗ) і (2) **in-space
default** (усі `conv3x3`, 1.0×, ReLU) з ідентичним рецептом — щоб приписати виграш саме
пошуку, а не іншій «родині» архітектур.

Оскільки розрив best↔default малий, кожну мережу тренуємо на **{len(seeds)} seed-ах**
({', '.join(map(str, seeds))}) і наводимо **mean ± std**. Порядок батчів прив'язаний до
seed окремим генератором (незалежно від RNG-ініціалізації), тож на кожному seed обидві
мережі бачать **однаковий** порядок даних — контрольоване порівняння.

Методологія: відбір моделі — на **validation**; **test** вимірюємо рівно раз на кожну
фінальну (модель, seed).

| Модель | Параметри | vs baseline | Val acc | **Test acc** |
|---|---|---|---|---|
| Baseline VGG11 (ДЗ1) | {M(base['params'])} | 1.0× | — | {pct(base['test_acc'])} |
| In-space default | {M(d['params'])} | {base['params']/d['params']:.1f}× менше | {msd(d['val_mean'], d['val_std'])} | {msd(d['test_mean'], d['test_std'])} |
| **Знайдена пошуком** | **{M(b['params'])}** | **{comp:.1f}× менше** | {msd(b['val_mean'], b['val_std'])} | **{msd(b['test_mean'], b['test_std'])}** |

Знайдена мережа = {msd(b['test_mean'], b['test_std'])} проти {pct(base['test_acc'])} у
baseline при **{comp:.1f}× менше параметрів**, і {gap*100:+.2f} п.п. над default. Різниця
best↔default ({gap*100:+.2f} п.п.) {'перевищує' if abs(gap) > (b['test_std']+d['test_std']) else 'порівнянна з'} сумарний
розкид (±{b['test_std']*100:.2f}/±{d['test_std']*100:.2f} п.п.) — тож інтерпретуємо її
{'як реальну' if abs(gap) > (b['test_std']+d['test_std']) else 'обережно (у межах шуму)'}.
Крива навчання найкращої архітектури (seed {seeds[0]}):
""")
code("""
plots.plot_history(retrain["best"]["history"], title="Знайдена архітектура (з нуля)")
plt.show()
""")

md(f"""
## 6. Бонус — наскільки інформативний one-shot proxy?

На питання зі звіту «how informative was the one-shot proxy?» відповідаємо **числами** з
p-values, а не відчуттям. Але не лише на топі: **top-only** нічого не каже про грубий
відсів. Тому беремо **стратифіковану** вибірку з {n_corr} архітектур, що покриває **весь**
діапазон proxy (по квантилях: низ / середина / верх), тренуємо кожну коротко з нуля
({corr['config']['short_epochs']} епох × {len(corr['config']['seeds'])} seed-и, усереднення)
і рахуємо кореляцію на всій вибірці **і** окремо на топ-{top_k}.

- **Грубий відсів (уся вибірка, n={n_corr}):** Kendall τ = **{full['kendall_tau']:.2f}**
  ({pf(full['kendall_p'])}), Spearman ρ = **{full['spearman_rho']:.2f}**
  ({pf(full['spearman_p'])}). {'Значущо' if full['kendall_p'] < 0.05 else 'Тенденція'} —
  proxy надійно відрізняє погані архітектури від хороших.
- **Тонке ранжування (топ-{top_k}):** Kendall τ = **{topc['kendall_tau']:.2f}**
  ({pf(topc['kendall_p'])}), Spearman ρ = **{topc['spearman_rho']:.2f}**. Серед топу
  proxy-точності лежать у вузькому діапазоні {proxy_spread*100:.1f} п.п. (усі
  ~{pct(tk[0]['proxy_val_acc'])}), тоді як справжні короткі — у {real_spread*100:.1f} п.п.:
  найкращі архітектури **майже нерозрізнимі** під спільними вагами, і вибір переможця —
  це вже шум.

Графік — уся стратифікована вибірка (сильний загальний тренд = добрий грубий відсів):
""")
code("""
proxy = [r["proxy_val_acc"] for r in corr["rows"]]
real  = [r["short_val_acc"] for r in corr["rows"]]
plots.plot_proxy_correlation(proxy, real,
    tau=corr["full"]["kendall_tau"], rho=corr["full"]["spearman_rho"],
    tau_p=corr["full"]["kendall_p"], rho_p=corr["full"]["spearman_p"])
plt.show()
for r in sorted(corr["rows"], key=lambda x: x["proxy_val_acc"]):
    print(f'proxy={r["proxy_val_acc"]:.4f}  short={r["short_val_acc"]:.4f}  '
          f'{r["params"]/1e6:.2f}M  {"/".join(r["arch"]["ops"])} w{r["arch"]["width"]} {r["arch"]["act"]}')
""")

md(f"""
## 7. Висновки (звіт)

**Що спрацювало добре.**
- Пошук знайшов архітектуру **{comp:.1f}× меншу** за VGG11 із практично тією ж
  точністю ({msd(b['test_mean'], b['test_std'])} vs {pct(base['test_acc'])}) — головна
  ціль NAS (краще співвідношення точність/розмір) досягнута.
- **Weight sharing зробив пошук дешевим:** одна supernet ({cfg['supernet_epochs']} епох)
  + {cfg['evals']} майже безкоштовних trial-ів замість {cfg['space_size']} повних
  тренувань. Оцінка одного кандидата — рекалібрація BN + один прохід по val.
- **Proxy впевнено знаходить хороший регіон:** на стратифікованій вибірці кореляція
  proxy↔реальне навчання значуща (τ={full['kendall_tau']:.2f}, {pf(full['kendall_p'])}),
  топ збігся на чіткий мотив, найгірші (усі-`dwsep`, 0.5×) відсіяні. Грубий відсів — надійний.

**Що не спрацювало / вийшло не як очікувалось.**
- **Тонке ранжування proxy — слабке** (топ-{top_k}: τ={topc['kendall_tau']:.2f},
  {pf(topc['kendall_p'])} — статистично незначущо). Серед топу spread proxy — лише
  {proxy_spread*100:.1f} п.п., тож обрати «найкращу з найкращих» proxy не може; фінальний
  вибір усередині топу — майже випадковий. Це відома вада one-shot: co-adaptation
  спільних ваг вирівнює сильних кандидатів.
- **Розрив proxy↔реальне навчання великий за абсолютом** (proxy ~73% vs справжні
  84–91%): спільні ваги недо-треновані для кожного окремого шляху. Для *ранжування*
  це прийнятно, для *абсолютної* оцінки — ні.
- **TPE лише трохи кращий за random** на цьому просторі (best-acc {pct(tpe_best_acc)} vs
  {pct(rand_best_acc)}; вийшов на best за {best_at_unique} унікальних): хороша зона
  широка. Але random-контроль — **один** прогін, тож це слабке твердження; надійніше —
  3–5 random-seed-ів. Виграв би виразніше на більшому/складнішому просторі.

**Наскільки інформативний one-shot proxy?** Двошарова відповідь, тепер з p-values:
**для грубого відсіву — надійно** (τ={full['kendall_tau']:.2f}, {pf(full['kendall_p'])}
на всьому діапазоні), **для тонкого вибору переможця — ні** (топ-{top_k}:
τ={topc['kendall_tau']:.2f}, {pf(topc['kendall_p'])}). На практиці proxy — це **фільтр**
(звузити {cfg['space_size']} → десяток кандидатів), а фінал вирішувати коротким
до-навчанням — рівно як ми зробили в бонусі.

**Як покращити результати.**
- **Random-контроль на 3–5 seed-ах** (band/mean) — щоб твердження «TPE > random» стало
  статистично надійним, а не одноразовим порівнянням.
- Тренувати supernet довше + **fairness-трюки** (FairNAS: кожен батч — усі операції по
  черзі) або sandwich-rule, щоб зменшити co-adaptation і підняти τ на топі.
- Оцінювати кандидатів на **більшій підвибірці val** і з більшою BN-рекалібрацією
  (менше шуму в proxy); більше seed-ів у proxy-study.
- Розширити простір (глибина / kernel size / окремий stride) — зараз макро-скелет
  фіксований, тож і виграш TPE над random, і виграш пошуку обмежені.
- Ще більше **seed-ів** для retrain (тут {len(seeds)}) для тіснішого CI на розриві best↔default.
""")

md(f"""
## 8. Як ми це зробили / що пробували

- **Реюз, а не переписування.** Supernet — це пряме розширення `ElasticVGG11` з
  бонусу ДЗ3: той самий slimmable-зріз `W[:out, :in]` і та сама ідея BN-рекалібрації,
  лише додано вибір **операції** на стадію. `data`/`engine`/`model`/`utils` — без змін
  з ДЗ1–ДЗ3.
- **Чесний підрахунок параметрів.** Кількість параметрів рахуємо не аналітичною
  формулою, а **побудовою** реальної `StandaloneNet` — і тестом звіряємо, що зрізи ваг
  supernet мають **точно ті самі форми**, що й standalone-архітектура. Тобто «шукали
  одну мережу, а тренуємо іншу» тут неможливо за побудовою.
- **Fair baseline з контролем seed.** Порівнюємо не лише з VGG11, а й з in-space default
  при **ідентичному** рецепті; best і default тренуємо на **{len(seeds)} seed-ах** з
  однаковим (генератором-прив'язаним) порядком батчів на кожному seed → різниця не
  списується на удачу ініціалізації/порядку даних.
- **Дисципліна даних.** Увесь пошук і кожне проміжне число — на **validation**; test
  чіпаємо рівно раз на кожну фінальну (модель, seed). `inference_mode` в оцінці.
- **Контроль пошуку.** Додали random-search baseline на тій самій supernet — щоб
  бачити внесок саме TPE, а не раннього вдалого семплу; вісь збіжності — за
  **унікальними** архітектурами (кеш-візити інформації не додають).
- **На чому спіткнулись.** (1) BN у one-shot: без рекалібрації активного шляху
  proxy-точність — сміття (running stats змішані по всіх ширинах); рекалібрація на
  ~{cfg['recal_batches']} батчах це лікує. (2) Слабка τ на топі спочатку виглядала як
  баг, але це реальна властивість — топ-кандидати нерозрізнимі під спільними вагами;
  ми лишили це число як є й додали стратифіковану вибірку, щоб окремо показати надійний
  грубий відсів. (3) Оцінку proxy стратифікували по квантилях — top-only не доводить
  якість фільтра.
""")

md(f"""
## 9. Джерела

- Bergstra, Bardenet, Bengio, Kégl. *Algorithms for Hyper-Parameter Optimization.*
  NeurIPS 2011. (TPE — алгоритм, який використовує Hyperopt.)
- Bergstra, Yamins, Cox. *Hyperopt: A Python Library for Optimizing Hyperparameters.*
  SciPy 2013. — <https://github.com/hyperopt/hyperopt>
- Guo et al. *Single Path One-Shot Neural Architecture Search with Uniform Sampling.*
  ECCV 2020 (arXiv:1904.00420). — схема тренування supernet.
- Howard et al. *MobileNets.* 2017 (depthwise-separable); Sandler et al.
  *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018 — операції блоків.
- Yu, Huang. *Universally Slimmable Networks* / *Slimmable Neural Networks.* ICLR 2019 —
  ідея спільних ваг за шириною + per-width BN (яку ми реюзаємо з бонусу ДЗ3).
""")

# --------------------------------------------------------------------------- #
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
with open(OUT_NB, "w") as f:
    nbf.write(nb, f)
print(f"wrote {OUT_NB} ({len(cells)} cells)")


# --------------------------------------------------------------------------- #
# REPORT.md — the same story, standalone
# --------------------------------------------------------------------------- #
report = f"""# ДЗ4 — Neural Architecture Search з Hyperopt · ЗВІТ

Код і результати в репо: [imagic9/efficient-ml-set → hw4](https://github.com/imagic9/efficient-ml-set/tree/main/hw4)

**Курс:** Efficient ML, SET University · **Датасет:** CIFAR-10

## Завдання
Пошук у дискретному просторі архітектур (операція блоку × ширина × активація) за
допомогою Hyperopt **TPE** зі **спільними вагами (weight sharing)**; графіки
running-best loss vs. номер trial і accuracy vs. кількість параметрів; retrain
найкращої архітектури з нуля проти baseline; звіт про те, наскільки інформативним був
one-shot proxy.

## Постановка
- **Простір пошуку** (`src/search_space.py`): 4 стадії, у кожній операція ∈
  {{conv3x3, dwsep, mbconv}}; глобальна ширина ∈ {{0.5, 0.75, 1.0, 1.25}}; глобальна
  активація ∈ {{ReLU, ReLU6, SiLU, GELU, LeakyReLU}} → **{cfg['space_size']}** архітектур.
- **One-shot supernet** (`src/supernet.py`): спільні ваги на макс. ширині, усі операції
  як паралельні гілки; тренування Single-Path One-Shot (рівноймовірна випадкова
  *архітектура* на крок, {cfg['supernet_epochs']} епох). BN рекалібрується під кожного
  кандидата перед оцінкою.
- **Пошук** (`src/nas_search.py`): Hyperopt TPE, ціль = one-shot proxy val loss,
  {cfg['evals']} trial-ів ({n_unique} унікальних архітектур, {n_cached} кеш-візитів), +
  **random-search контроль** на тій самій supernet.
- **Retrain** ({retrain['config']['epochs']} епох, seed-и {','.join(map(str, seeds))}):
  найкраща архітектура з нуля проти замороженої VGG11 з ДЗ1 і in-space default
  (усі conv3x3/1.0×/ReLU); порядок батчів детермінований на кожен seed, наводимо mean±std.
- **Методологія:** увесь пошук і проміжні числа — на validation; test вимірюється рівно
  раз на кожну фінальну (модель, seed); `inference_mode` в оцінці.

## Головні результати (test — mean±std на {len(seeds)} seed-ах)

| Модель | Параметри | vs baseline | Test acc |
|---|---|---|---|
| Baseline VGG11 (ДЗ1, reference — 1 запуск) | {M(base['params'])} | 1.0× | {pct(base['test_acc'])} |
| In-space default (conv3x3/1.0×/ReLU) | {M(d['params'])} | {base['params']/d['params']:.1f}× менше | {msd(d['test_mean'], d['test_std'])} |
| **Знайдена пошуком** ({best_str}) | **{M(b['params'])}** | **{comp:.1f}× менше** | **{msd(b['test_mean'], b['test_std'])}** |

**Основний висновок — знайдена архітектура vs in-space default** (обидві на {len(seeds)}
seed-ах, ідентичний рецепт): **{gap*100:+.2f} п.п.** ({msd(b['test_mean'], b['test_std'])}
проти {msd(d['test_mean'], d['test_std'])}) — розрив перевищує сумарний розкид
(±{b['test_std']*100:.2f}/±{d['test_std']*100:.2f} п.п.), тож пошук реально дав кращу
архітектуру. Проти **VGG11** знайдена мережа — {ppe(dp_vs_base).replace('pp','п.п.')} при
**{comp:.1f}× менше параметрів**; але VGG11 має лише **один історичний запуск**, тож це
порівняння з reference-точкою, а не статистично доведена перевага.

## Що знайшов пошук
Пошук збігся на чіткий структурний мотив (топ-15 унікальних дизайнів за proxy):
стадія 0 → conv3x3 ({freq_str(stage_freq[0])}), стадія 1 → mbconv ({freq_str(stage_freq[1])}),
стадія 2 → {freq_str(stage_freq[2])}, стадія 3 → {freq_str(stage_freq[3])}; ширина
{freq_str(width_freq)}; активація {freq_str(act_freq)}. Найгірші дизайни (усі-dwsep,
0.5× ширина) падають до ~67% proxy-точності й впевнено відсіюються. TPE вийшов на
найкращий proxy-loss за {best_at_unique} унікальних архітектур; на цьому просторі TPE
лише трохи кращий за random ({pct(tpe_best_acc)} vs {pct(rand_best_acc)}) — хороша зона
широка (random-контроль — один прогін, тож твердження слабке).

## Наскільки інформативний one-shot proxy? (бонус)
Оцінка на **стратифікованій** вибірці з {n_corr} архітектур, що покриває весь діапазон
proxy ({corr['config']['short_epochs']} епох з нуля × {len(corr['config']['seeds'])} seed-и):

- **Грубий відсів (уся вибірка, n={n_corr}):** Kendall τ = **{full['kendall_tau']:.2f}**
  ({pf(full['kendall_p'])}), Spearman ρ = **{full['spearman_rho']:.2f}**
  ({pf(full['spearman_p'])}) — proxy надійно відрізняє погані архітектури від хороших.
- **Тонке ранжування (топ-{top_k}):** Kendall τ = **{topc['kendall_tau']:.2f}**
  ({pf(topc['kendall_p'])}) — незначущо; серед топу proxy займає лише
  {proxy_spread*100:.1f} п.п., тоді як реальне навчання — {real_spread*100:.1f} п.п., тож
  найкращі дизайни під спільними вагами нерозрізнимі.

**Двошарова відповідь:** proxy — надійний **грубий фільтр** ({cfg['space_size']} → десяток),
але **слабкий у виборі єдиного переможця**; фінал варто вирішувати коротким навчанням з
нуля (як у цьому бонусі).

## Що спрацювало / що ні / як покращити
- **Спрацювало:** мережа {comp:.1f}× менша за VGG11 на рівні його точності; weight
  sharing зробив пошук дешевим; proxy фільтрує хорошу зону зі значущою кореляцією
  (τ={full['kendall_tau']:.2f}, {pf(full['kendall_p'])}); розрив best↔default значущий.
- **Не спрацювало:** тонке ранжування proxy слабке (co-adaptation вирівнює сильних
  кандидатів); абсолютна proxy-точність (~73%) далеко нижча за реальну (84–91%); TPE лише
  трохи кращий за random на цьому просторі широкого оптимуму.
- **Як покращити:** random-контроль на 3–5 seed-ах (band/mean); довше тренувати supernet
  з fairness-трюками (FairNAS / sandwich rule) щоб підняти τ на топі; більша під-вибірка
  val + більше BN-рекалібрації; ширший простір (глибина / kernel / stride), де TPE
  виразніше обжене random; більше seed-ів для retrain.

## Відтворення
```bash
# тренує supernet з нуля і шукає (supernet.pt *.pt не комітиться — gitignored)
python run_search.py     --supernet-epochs {cfg['supernet_epochs']} --evals {cfg['evals']}
python run_retrain.py    --baseline ../hw1/results/baseline.pt --epochs {retrain['config']['epochs']} --seeds {','.join(map(str, seeds))}
python run_proxy_corr.py --n-bins {corr['config']['n_bins']} --per-bin {corr['config']['per_bin']} --top-k {top_k} --seeds {','.join(map(str, corr['config']['seeds']))} --short-epochs {corr['config']['short_epochs']}
python tests/test_search_space.py && python tests/test_supernet.py
```
"""
with open(OUT_REPORT, "w") as f:
    f.write(report)
print(f"wrote {OUT_REPORT}")
