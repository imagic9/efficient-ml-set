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

cfg = search["config"]
records = search["records"]
best = search["best_arch"]
best_str = "/".join(best["ops"]) + f", {best['width']}×, {best['act']}"

# distinct architectures actually evaluated
seen = {}
for r in records:
    k = (tuple(r["arch"]["ops"]), r["arch"]["width"], r["arch"]["act"])
    seen.setdefault(k, r)
n_unique = len(seen)
n_cached = sum(1 for r in records if r["cached"])
uniq_sorted = sorted(seen.values(), key=lambda r: r["val_acc"], reverse=True)
top15 = uniq_sorted[:15]

# operation / width / activation frequency across the top-15 distinct designs
stage_freq = [Counter(r["arch"]["ops"][s] for r in top15) for s in range(4)]
width_freq = Counter(r["arch"]["width"] for r in top15)
act_freq = Counter(r["arch"]["act"] for r in top15)
freq_str = lambda c: ", ".join(f"{k}×{v}" for k, v in c.most_common())

base = retrain["baseline_vgg11"]
b = retrain["best"]
d = retrain["default"]
comp = base["params"] / b["params"]
dp_vs_base = b["test_acc"] - base["test_acc"]
dp_vs_default = b["test_acc"] - d["test_acc"]

tau, rho = corr["kendall_tau"], corr["spearman_rho"]
crows = corr["rows"]
proxy_spread = max(r["proxy_val_acc"] for r in crows) - min(r["proxy_val_acc"] for r in crows)
real_spread = max(r["short_val_acc"] for r in crows) - min(r["short_val_acc"] for r in crows)

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

### Головний результат (тест — раз на фінальну модель)

| Модель | Параметри | Test acc |
|---|---|---|
| Baseline VGG11 (з ДЗ1) | {M(base['params'])} | {pct(base['test_acc'])} |
| In-space default (усі conv3x3, 1.0×, ReLU) | {M(d['params'])} | {pct(d['test_acc'])} |
| **Знайдена пошуком** ({best_str}) | **{M(b['params'])}** | **{pct(b['test_acc'])}** |

Знайдена архітектура дає **{pct(b['test_acc'])}** — практично рівень baseline
({pp(dp_vs_base)}) при **{comp:.1f}× менше параметрів** ({M(b['params'])} проти
{M(base['params'])}), і на {pp(dp_vs_default)} краще за розумний default у тому ж
просторі. Тобто пошук знайшов набагато **компактнішу** мережу без втрати точності.

Наскільки надійним був one-shot proxy? Kendall τ = **{tau:.2f}**, Spearman ρ =
**{rho:.2f}** — тобто **грубо** proxy знаходить хороший регіон простору бездоганно
(усі топ-архітектури мають один мотив), але **тонко** ранжувати найкращих між собою
він майже не вміє. Деталі — у розділах 4–6 і у висновках.
""")

md("## 1. Підготовка")
code("""
import json
import torch
import matplotlib.pyplot as plt

from src.utils import get_device
from src.search_space import space_size, count_arch_params, StandaloneNet, sample_arch
from src import plots

RESULTS = "results"
def load(n):
    with open(f"{RESULTS}/{n}") as f: return json.load(f)
search   = load("search.json")
retrain  = load("retrain.json")
corr     = load("proxy_corr.json")
records  = search["records"]
print("device:", get_device(), "| простір:", space_size(), "архітектур")
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
оновлюємо лише її шлях. За багато кроків усі операції/ширини/активації отримують
градієнти, і жоден шлях не має апріорної переваги.

Statistics BatchNorm у one-shot ненадійні (вони змішують усі ширини/операції, які
бачили під час тренування), тому **перед кожною оцінкою** ми **рекалібруємо** BN
активного шляху на кількох батчах — стандартна практика one-shot NAS.

Supernet тут — {M(search['supernet_params'])} параметрів (max ширина, усі операції),
тренували {cfg['supernet_epochs']} епох. Крива нижче — середня точність випадкових
шляхів по епохах (не одна модель, а розподіл над шляхами — тож це грубий сигнал
«здоров'я» supernet, а не фінальна точність).
""")
code("""
h = search["supernet_history"]
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(range(1, len(h["train_acc"])+1), [a*100 for a in h["train_acc"]], color="tab:purple")
ax.set_xlabel("епоха"); ax.set_ylabel("сер. точність шляху, %")
ax.set_title("SPOS-тренування supernet (середнє по випадкових шляхах)")
ax.grid(alpha=0.3); plt.show()
print(f"фінальна сер. точність шляху: {h['train_acc'][-1]*100:.1f}%")
""")

md(f"""
## 4. Пошук TPE (weight-sharing proxy)

TPE будує ймовірнісну модель «які конфігурації дають низький лос» і пропонує нові
кандидати, що максимізують очікуване покращення. Кожен trial коштує копійки: вибрати
під-шлях у натренованій supernet → рекалібрувати BN → порахувати **validation loss**.
Це proxy-оцінка; ми також фіксуємо кількість параметрів і proxy val-accuracy.

За {cfg['evals']} trial-ів TPE оцінив **{n_unique}** унікальних архітектур (з
{cfg['space_size']}); решта {n_cached} — повторні візити вже баченого (TPE
експлуатує вдалі зони — беремо їх з кешу).

**Два обов'язкові графіки:**
""")
code("""
plots.plot_search_convergence(records); plt.show()
plots.plot_acc_vs_params(records, search["best_arch"]); plt.show()
""")

md(f"""
**Що знайшов пошук.** Топ-архітектури за proxy напрочуд узгоджені — TPE збігся на
чіткий структурний мотив:

- **стадія 0 → `conv3x3`** ({freq_str(stage_freq[0])}): перший шар з «сирих» 3 каналів
  хоче повну згортку;
- **стадія 1 → `mbconv`** ({freq_str(stage_freq[1])}): inverted residual у середині;
- стадія 2 → {freq_str(stage_freq[2])}; стадія 3 → {freq_str(stage_freq[3])};
- ширина: {freq_str(width_freq)} (ніколи 0.5× у топі); активація: {freq_str(act_freq)}.

Найгірші архітектури — усі-`dwsep` при ширині 0.5× — падають до ~67% proxy-точності;
proxy впевнено відсіює їх. **Найкраща за proxy:** `{best_str}`, {M(search['best_params'])},
proxy val-acc = {pct(search['best_proxy_val_acc'])}.
""")

md(f"""
## 5. Retrain з нуля vs baseline

Proxy лише **ранжує**; фінальне число дає навчання обраної архітектури **з нуля** як
звичайної мережі ({cfg['seed']}-seed, {retrain['config']['epochs']} епох, cosine LR).
Порівнюємо з двома baseline: (1) заморожена VGG11 з ДЗ1 (той самий baseline через усі
ДЗ) і (2) **in-space default** (усі `conv3x3`, 1.0×, ReLU) з ідентичним рецептом — щоб
приписати виграш саме пошуку, а не іншій «родині» архітектур.

Методологія: відбір моделі — на **validation**; **test** вимірюємо рівно раз на кожну
фінальну модель.

| Модель | Параметри | vs baseline | Val acc | **Test acc** |
|---|---|---|---|---|
| Baseline VGG11 (ДЗ1) | {M(base['params'])} | 1.0× | — | {pct(base['test_acc'])} |
| In-space default | {M(d['params'])} | {base['params']/d['params']:.1f}× менше | {pct(d['val_acc'])} | {pct(d['test_acc'])} |
| **Знайдена пошуком** | **{M(b['params'])}** | **{comp:.1f}× менше** | {pct(b['val_acc'])} | **{pct(b['test_acc'])}** |

Знайдена мережа = {pct(b['test_acc'])} проти {pct(base['test_acc'])} у baseline
({pp(dp_vs_base)}) при **{comp:.1f}× менше параметрів**, і {pp(dp_vs_default)} над
default. Крива навчання найкращої архітектури:
""")
code("""
plots.plot_history(retrain["best"]["history"], title="Знайдена архітектура (з нуля)")
plt.show()
""")

md(f"""
## 6. Бонус — наскільки інформативний one-shot proxy?

На питання зі звіту «how informative was the one-shot proxy?» відповідаємо **числом**,
а не відчуттям. Беремо топ-{corr['config']['top_k']} архітектур за proxy, тренуємо
кожну **коротко з нуля** ({corr['config']['short_epochs']} епох) і міряємо, наскільки
proxy-ранжування збігається з ранжуванням «справжнього» короткого навчання —
через Kendall τ і Spearman ρ.

**Результат: Kendall τ = {tau:.2f}, Spearman ρ = {rho:.2f}** — слабка додатна
кореляція. Причина видно з чисел: серед топ-{corr['config']['top_k']} proxy-точності
лежать у вузькому діапазоні {proxy_spread*100:.1f} п.п. (усі ~{pct(crows[0]['proxy_val_acc'])}),
тоді як справжні короткі — у {real_spread*100:.1f} п.п. Тобто під спільними вагами
найкращі архітектури **майже нерозрізнимі**, і тонке ранжування між ними — це шум.
""")
code("""
proxy = [r["proxy_val_acc"] for r in corr["rows"]]
real  = [r["short_val_acc"] for r in corr["rows"]]
plots.plot_proxy_correlation(proxy, real, tau=corr["kendall_tau"], rho=corr["spearman_rho"])
plt.show()
for r in corr["rows"]:
    print(f'proxy={r["proxy_val_acc"]:.4f}  short={r["short_val_acc"]:.4f}  '
          f'{r["params"]/1e6:.2f}M  {"/".join(r["arch"]["ops"])} w{r["arch"]["width"]} {r["arch"]["act"]}')
""")

md(f"""
## 7. Висновки (звіт)

**Що спрацювало добре.**
- Пошук знайшов архітектуру **{comp:.1f}× меншу** за VGG11 із практично тією ж
  точністю ({pct(b['test_acc'])} vs {pct(base['test_acc'])}) — головна ціль NAS
  (краще співвідношення точність/розмір) досягнута.
- **Weight sharing зробив пошук дешевим:** одна supernet ({cfg['supernet_epochs']} епох)
  + {cfg['evals']} майже безкоштовних trial-ів замість {cfg['space_size']} повних
  тренувань. Оцінка одного кандидата — рекалібрація BN + один прохід по val.
- **TPE + proxy впевнено знаходять хороший регіон:** топ збігся на чіткий мотив
  (`conv3x3` на вході, `mbconv` в середині, повна ширина, ReLU6), а найгірші
  (усі-`dwsep`, 0.5×) відсіяні. Грубе ранжування — надійне.

**Що не спрацювало / вийшло не як очікувалось.**
- **Тонке ранжування proxy — слабке** (τ={tau:.2f}). Серед топ-архітектур spread
  proxy — лише {proxy_spread*100:.1f} п.п., тож обрати «найкращу з найкращих» proxy не
  може; фінальний вибір усередині топу — майже випадковий (нам пощастило, що обрана
  архітектура добре до-тренувалась). Це відома вада one-shot: co-adaptation спільних
  ваг вирівнює сильних кандидатів.
- **Разрив proxy↔реальне навчання великий за абсолютом** (proxy ~73% vs справжні
  84–91%): спільні ваги недо-треновані для кожного окремого шляху. Для *ранжування*
  це прийнятно, для *абсолютної* оцінки — ні.

**Наскільки інформативний one-shot proxy?** Двошарова відповідь: **для грубого
відсіву — дуже** (безпомилково відкидає слабкі операції/ширини й знаходить хорошу
зону), **для тонкого вибору переможця — слабко** (τ={tau:.2f}). На практиці proxy варто
використовувати як **фільтр** (звузити 1620 → десяток кандидатів), а фінал вирішувати
коротким справжнім до-навчанням — рівно як ми зробили в бонусі.

**Як покращити результати.**
- Тренувати supernet довше + **fairness-трюки** (FairNAS: кожен батч — усі операції по
  черзі) або sandwich-rule, щоб зменшити co-adaptation і підняти τ.
- Оцінювати кандидатів на **більшій підвибірці val** і з більшою BN-рекалібрацією
  (менше шуму в proxy).
- Розширити простір (глибина/kernel size/окремий stride) — зараз макро-скелет
  фіксований, тож виграш обмежений.
- Багато **seed-ів** і mean±std (усі числа тут — single-seed) для статистично
  надійних висновків.
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
- **Fair baseline.** Порівнюємо не лише з VGG11, а й з in-space default при
  **ідентичному** рецепті тренування — інакше виграш можна було б списати на кращий
  розклад/епохи, а не на архітектуру.
- **Дисципліна даних.** Увесь пошук і кожне проміжне число — на **validation**; test
  чіпаємо рівно раз на фінальну модель (best / default / baseline). `inference_mode`
  в оцінці.
- **На чому спіткнулись.** (1) BN у one-shot: без рекалібрації активного шляху
  proxy-точність — сміття (running stats змішані по всіх ширинах); рекалібрація на
  ~{cfg['recal_batches']} батчах це лікує. (2) Слабка τ спочатку виглядала як баг, але
  це реальна властивість — топ-кандидати нерозрізнимі під спільними вагами; ми лишили
  це число як є, а не «покращили» його підбором. (3) Числа — single-seed: позначено
  скрізь.
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
report = f"""# HW4 — Neural Architecture Search with Hyperopt · REPORT

**Course:** Efficient ML, SET University · **Dataset:** CIFAR-10

## Task
Search a discrete space of architectures (block op × width × activation) with
Hyperopt **TPE**, using **weight sharing**; plot running-best loss vs. trial and
accuracy vs. #params; retrain the best design from scratch vs. a baseline; report
how informative the one-shot proxy was.

## Setup
- **Search space** (`src/search_space.py`): 4 stages, each a block op ∈
  {{conv3x3, dwsep, mbconv}}; global width ∈ {{0.5, 0.75, 1.0, 1.25}}; global
  activation ∈ {{ReLU, ReLU6, SiLU, GELU, LeakyReLU}} → **{cfg['space_size']}** archs.
- **One-shot supernet** (`src/supernet.py`): shared weights at max width, all ops as
  parallel branches, trained with Single-Path One-Shot (uniform random path/step,
  {cfg['supernet_epochs']} epochs). Per-candidate BN recalibrated before scoring.
- **Search** (`src/nas_search.py`): Hyperopt TPE, objective = one-shot proxy val loss,
  {cfg['evals']} trials ({n_unique} unique archs, {n_cached} cached revisits).
- **Retrain** ({retrain['config']['epochs']} epochs, seed {cfg['seed']}): best design
  from scratch vs. frozen HW1 VGG11 and an in-space default (all conv3x3/1.0×/ReLU).
- **Methodology:** all search/intermediate numbers on validation; test measured once
  per final model; single-seed (labelled as such).

## Headline results (test — once per model)

| Model | Params | vs baseline | Test acc |
|---|---|---|---|
| Baseline VGG11 (HW1) | {M(base['params'])} | 1.0× | {pct(base['test_acc'])} |
| In-space default (conv3x3/1.0×/ReLU) | {M(d['params'])} | {base['params']/d['params']:.1f}× smaller | {pct(d['test_acc'])} |
| **Found by search** ({best_str}) | **{M(b['params'])}** | **{comp:.1f}× smaller** | **{pct(b['test_acc'])}** |

The searched design reaches **{pct(b['test_acc'])}** — essentially baseline accuracy
({ppe(dp_vs_base)}) at **{comp:.1f}× fewer parameters**, and {ppe(dp_vs_default)} over
the in-space default.

## What the search found
TPE converged on a clear structural motif (top-15 distinct designs by proxy):
stage 0 → conv3x3 ({freq_str(stage_freq[0])}), stage 1 → mbconv ({freq_str(stage_freq[1])}),
stage 2 → {freq_str(stage_freq[2])}, stage 3 → {freq_str(stage_freq[3])}; width
{freq_str(width_freq)}; activation {freq_str(act_freq)}. The worst designs (all-dwsep,
0.5× width) collapse to ~67% proxy accuracy and are reliably rejected.

## How informative was the one-shot proxy? (bonus)
Top-{corr['config']['top_k']} by proxy, each trained {corr['config']['short_epochs']}
epochs from scratch: **Kendall τ = {tau:.2f}, Spearman ρ = {rho:.2f}** (weak positive).
Among the top designs the proxy spans only {proxy_spread*100:.1f} pp while real
short-training spans {real_spread*100:.1f} pp — under shared weights the best
architectures are nearly indistinguishable, so fine ranking is noise.

**Two-part answer:** the proxy is **very** informative for *coarse filtering* (it nails
the good region and discards weak ops/widths) but **weak** for *picking the single
winner* (τ={tau:.2f}). Best used as a filter (1620 → a dozen), with the final choice
decided by short honest from-scratch training.

## What worked / what didn't / how to improve
- **Worked:** {comp:.1f}× smaller net at baseline accuracy; weight sharing made the
  search cheap; TPE + proxy find the good region confidently.
- **Didn't:** fine proxy ranking is weak (co-adaptation of shared weights flattens
  strong candidates); absolute proxy accuracy (~73%) is far below real (84–91%).
- **Improve:** train the supernet longer with fairness tricks (FairNAS / sandwich
  rule); larger val subset + more BN recal batches; widen the space (depth / kernel /
  stride); multi-seed mean±std (all numbers here are single-seed).

## Reproduce
```bash
python run_search.py     --supernet-epochs {cfg['supernet_epochs']} --evals {cfg['evals']}
python run_retrain.py    --baseline ../hw1/results/baseline.pt --epochs {retrain['config']['epochs']}
python run_proxy_corr.py --top-k {corr['config']['top_k']} --short-epochs {corr['config']['short_epochs']}
python tests/test_search_space.py && python tests/test_supernet.py
```
"""
with open(OUT_REPORT, "w") as f:
    f.write(report)
print(f"wrote {OUT_REPORT}")
