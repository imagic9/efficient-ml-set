"""Assemble the submission notebook (HW3_Self_Distillation.ipynb) and REPORT.md.

Reads metrics produced by run_distill.py + run_kd_ablation.py from RESULTS_DIR and
bakes the real numbers into a Ukrainian narrative. Run AFTER both experiments
finish, then execute the notebook with nbconvert.
"""
import json
import os
import sys

import nbformat as nbf

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT_NB = "HW3_Self_Distillation.ipynb"
OUT_REPORT = "REPORT.md"


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


dist = load("distill.json")
abl = load("kd_ablation.json")
try:
    na = load("netaug.json")          # bonus (optional -- core builds without it)
except FileNotFoundError:
    na = None

teacher = dist["teacher"]
cfg = dist["config"]
REGIME_ORDER = ["pruned", "quant2", "prune_quant"]
regimes = [r for r in REGIME_ORDER if r in dist]

pct = lambda x: f"{x * 100:.2f}%"
pp = lambda x: f"{x * 100:+.2f} п.п."

# per-regime deltas
def d(r):
    x = dist[r]
    return {
        "tag": x["tag"], "size_MB": x["size_MB"], "comp": x["compression_x"],
        "pre": x["pre_finetune_val"],
        "ce_val": x["ce"]["val"], "ce_test": x["ce"]["test"],
        "kd_val": x["kd"]["val"], "kd_test": x["kd"]["test"],
        "dtest": x["kd"]["test"] - x["ce"]["test"],
        "dval": x["kd"]["val"] - x["ce"]["val"],
    }

R = {r: d(r) for r in regimes}
# the regime where KD helps the most on test (the headline of the story)
best_r = max(regimes, key=lambda r: R[r]["dtest"])
# quant2 is the hardest / most illustrative case
hard = "quant2" if "quant2" in R else best_r

# ablation-derived
temp_pts = abl["temperature_sweep"]
alpha_pts = abl["alpha_sweep"]
ce_ref = abl.get("ce_only_val")
best_T = max(temp_pts, key=lambda p: p[1])
best_a = max(alpha_pts, key=lambda p: p[1])
# alpha=0 (pure KD) vs alpha=1 (pure CE) endpoints, if present
a_pure_kd = next((v for a, v in alpha_pts if a == 0.0), None)
a_pure_ce = next((v for a, v in alpha_pts if a == 1.0), None)

cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))


# --------------------------------------------------------------------------- #
def summary_rows():
    rows = []
    for r in regimes:
        x = R[r]
        rows.append(
            f"| {x['tag']} | {x['comp']:.1f}× | {x['size_MB']:.2f} | "
            f"{pct(x['ce_test'])} | **{pct(x['kd_test'])}** | {pp(x['dtest'])} |")
    return "\n".join(rows)


md(f"""
# Домашня робота 3 — Self-Distillation зі стиснутими моделями (VGG11 / CIFAR-10)

**Курс:** Efficient ML, SET University

Ідея knowledge distillation (KD): маленька/стиснута модель («учень») вчиться не лише
на жорстких мітках, а й на **м'яких ймовірностях великої моделі-«вчителя»** — тих
самих «майже правильних» відтінках (кіт трохи схожий на собаку й зовсім не на
вантажівку), яких у one-hot мітках немає. Тут це **self-distillation**: вчитель — наша
ж непорушена fp32 VGG11 з ДЗ1/ДЗ2 ({pct(teacher['test'])} на тесті), а учень — та сама
архітектура, але **стиснута** інструментами ДЗ1 (прунінг) і ДЗ2 (квантизація).

**Функція втрат** (Hinton, 2015):

$$L = \\alpha\\,\\mathrm{{CE}}(\\text{{student}}, y) \\;+\\; (1-\\alpha)\\,T^2\\,
\\mathrm{{KL}}\\!\\left(\\text{{softmax}}(\\tfrac{{z_t}}{{T}}) \\,\\big\\|\\,
\\text{{softmax}}(\\tfrac{{z_s}}{{T}})\\right)$$

Температура $T>1$ «розм'якшує» обидва розподіли, множник $T^2$ повертає масштаб
градієнта м'якої частини (він інакше падає як $1/T^2$), а $\\alpha$ балансує жорстку і
м'яку складові: $\\alpha=1$ — чистий CE (наш baseline відновлення), $\\alpha=0$ — чиста
дистиляція.

**Головне порівняння — чесна пара KD vs CE.** Для кожного стиснутого учня ми
до-навчаємо його **двічі з однакового стиснутого старту**, з ідентичними
оптимізатором, розкладом LR, кількістю епох і seed. Одна гілка — звичайне відновлення
на CE (чесний baseline), друга додає м'які таргети вчителя. Тож будь-яка різниця в
точності — заслуга саме дистиляції.

### Підсумок (тест — раз на модель, T={cfg['T']:.0f}, α={cfg['alpha']})

| Стиснутий учень | Стиснення | Розмір, МБ | CE-only (тест) | KD (тест) | KD − CE |
|---|---|---|---|---|---|
| Вчитель fp32 (з ДЗ1) | 1× | {teacher['size_MB']:.2f} | — | {pct(teacher['test'])} | — |
{summary_rows()}

Виграш дистиляції **скромний**; у цьому запуску простежується **тенденція** — що
сильніше стиснення, то більший виграш: від {pp(R['pruned']['dtest']) if 'pruned' in R else '—'}
на найлегшому учні ({R['pruned']['comp']:.1f}×) до {pp(R[best_r]['dtest'])} на
найагресивнішому ({R[best_r]['tag']}, {R[best_r]['comp']:.1f}×). Числа — single-seed (тест
раз на модель), тож це тенденція, а не строгий закон. Там, де CE-only і сам майже
дотягується до вчителя (прунінг), м'яким таргетам нема куди тягнути; що більше точності
з'їдає стиснення — то більше простору для KD. Найкраща точка сумарно —
{R[best_r]['tag']}: {pct(R[best_r]['kd_test'])} при {R[best_r]['comp']:.1f}× стисненні.
""")

md("""
## 1. Підготовка
""")
code("""
import json, inspect
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, get_device
from src.data import build_loaders
from src.model import build_vgg11_cifar, count_parameters
from src import distill, prune, kmeans_quant, qat, plots

RESULTS = "results"
set_seed(42)
device = get_device()
print("device:", device, "| torch:", torch.__version__)
""")

md(f"""
## 2. Дані та вчитель

CIFAR-10, той самий поділ, що й у ДЗ1/ДЗ2: 45 000 train / 5 000 val / 10 000 test.
Валідація — для всіх проміжних рішень (криві відновлення, ablation), тест — рівно один
раз на фінальну модель. **Вчитель** — щільна VGG11 з ДЗ1 (fp32, {pct(teacher['test'])}
на тесті, {teacher['size_MB']:.2f} МБ), заморожена (`eval()`, без градієнтів).
""")
code("""
train_loader, val_loader, test_loader = build_loaders("./data", batch_size=256)
print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

teacher = build_vgg11_cifar().to(device)
teacher.load_state_dict(torch.load("../hw1/results/baseline.pt", map_location=device))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"вчитель: {count_parameters(teacher)/1e6:.2f}M параметрів")
""")

md("""
## 3. KD-loss: жорсткі + м'які мітки

Ось уся дистиляційна логіка. `alpha=1` вимикає KL-складову й повертає чистий
`CrossEntropyLoss` — тому і KD-, і CE-only-прогони йдуть **одним кодом**, різниця лише
в `alpha`. Множник `T²` тримає градієнт м'якої частини на тому самому масштабі, що й
CE, тож один learning rate працює за будь-якої температури.
""")
code("""
print(inspect.getsource(distill.DistillLoss))
""")
md("""
Швидка перевірка властивостей (те саме, що в `tests/test_distill.py`): при `alpha=1`
KD-loss дорівнює крос-ентропії, а якщо учень уже збігається з учителем — чиста
KD-складова (`alpha=0`) прямує до нуля.
""")
code("""
import torch.nn.functional as F
s = torch.randn(16, 10, requires_grad=True); t = torch.randn(16, 10)
y = torch.randint(0, 10, (16,))
print("alpha=1 == CE:", torch.allclose(distill.DistillLoss(alpha=1.0)(s, t, y), F.cross_entropy(s, y)))
z = torch.randn(16, 10)
print("matched logits -> ~0 KD:", distill.DistillLoss(alpha=0.0)(z.clone().requires_grad_(True), z.clone(), y).item())
""")

# --------------------------------------------------------------------------- #
md(f"""
## 4. Головний експеримент — KD vs CE для трьох стиснутих учнів

Три режими стиснення (все — та сама VGG11):

- **{R['pruned']['tag'] if 'pruned' in R else '—'}** — unstructured magnitude pruning з
  ДЗ1; маски накладаються після кожного кроку, відновлення SGD+cosine.
- **{R['quant2']['tag'] if 'quant2' in R else '—'}** — найважчий випадок (лише 4 значення
  ваги на шар); відновлення — centroid-QAT з ДЗ2 під KD-лоссом замість CE.
- **{R['prune_quant']['tag'] if 'prune_quant' in R else '—'}** — учень «Deep Compression»:
  спершу прунінг, потім квантизація ненульових ваг, відновлення centroid-QAT + KD.

Для кожного — дві гілки з однакового старту й seed: CE-only і KD. Дивимось на різницю.
""")
code("""
dist = json.load(open(f"{RESULTS}/distill.json"))
order = [r for r in ["pruned", "quant2", "prune_quant"] if r in dist]
tags = [dist[r]["tag"] for r in order]
ce_t = [dist[r]["ce"]["test"] for r in order]
kd_t = [dist[r]["kd"]["test"] for r in order]
plots.plot_kd_comparison(tags, ce_t, kd_t, baseline_acc=dist["teacher"]["test"],
                         title="KD vs CE-only відновлення (тест)")
plt.show()
print("TEST (раз на модель):")
for r in order:
    x = dist[r]
    print(f"  {x['tag']:<16}: init val {x['pre_finetune_val']*100:5.2f}%  |  "
          f"CE {x['ce']['test']*100:5.2f}%  KD {x['kd']['test']*100:5.2f}%  "
          f"({(x['kd']['test']-x['ce']['test'])*100:+.2f} pp)  @ {x['size_MB']:.2f}MB ({x['compression_x']:.1f}x)")
""")

md(f"""
Криві відновлення на найважчому учні ({R[hard]['tag']}): PTQ-старт валиться до
{pct(R[hard]['pre'])} на val, а до-навчання центроїдів під учителя піднімає його назад.
""")
code(f"""
hist = dist["{hard}"]["kd"]["history"]
plots.plot_history(hist, "KD відновлення — {R[hard]['tag']}")
plt.show()
""")

md(f"""
**Що видно.** KD і CE обидва добре відновлюють точність, а виграш дистиляції у цьому
запуску **тим більший, чим сильніше стиснення** (тенденція single-seed, у порядку
зростання стиснення):

- {R['pruned']['tag'] if 'pruned' in R else '—'} ({R['pruned']['comp']:.1f}×): {pp(R['pruned']['dtest']) if 'pruned' in R else '—'} на тесті —
  80%-розріджена модель зберігає fp32-точність ненульових ваг, тож CE й сам майже
  дотягується до вчителя, і м'яким таргетам мало що додати.
- {R['quant2']['tag'] if 'quant2' in R else '—'} ({R['quant2']['comp']:.1f}×): {pp(R['quant2']['dtest']) if 'quant2' in R else '—'} — лише 4 значення ваги
  на шар, розрив після стиснення великий, і «темні знання» вчителя вже помітно допомагають.
{('- ' + R['prune_quant']['tag'] + ' (' + f"{R['prune_quant']['comp']:.1f}×): " + pp(R['prune_quant']['dtest']) + ' — найсильніше стиснення дає й найбільший виграш KD; поєднання прунінгу і квантизації — найкраща точка сумарно (' + pct(R['prune_quant']['kd_test']) + ').') if 'prune_quant' in R else ''}

Величина виграшу невелика (десяті частки п.п.) — це чесний результат, а не помилка:
причини розбираємо у звіті нижче (учитель сам лише {pct(teacher['test'])}, self-distillation
з однаковою архітектурою має обмежену «стелю»).
""")

# --------------------------------------------------------------------------- #
md(f"""
## 5. Ablation — дві ручки KD (учень {abl.get('tag', '2-bit')}, val)

Свіпи робимо на **2-бітному** учні (найбільший розрив → найбільше простору для KD) і
**лише на валідації** — це підбір гіперпараметрів, торкатися тесту тут не можна.

- **Температура T** (α={abl['fixed_alpha']} фіксовано): скільки «пом'якшувати» розподіл
  вчителя.
- **α** (T={abl['fixed_temp']:.0f} фіксовано): баланс CE ↔ м'які мітки; α=1 — чистий CE
  (референсна лінія).
""")
code("""
abl = json.load(open(f"{RESULTS}/kd_ablation.json"))
tp = abl["temperature_sweep"]; ap = abl["alpha_sweep"]
plots.plot_kd_ablation([t for t, _ in tp], [v for _, v in tp],
                       xlabel="температура T", baseline_val=abl["teacher_val"],
                       ce_val=abl.get("ce_only_val"),
                       title=f"2-bit учень: свіп температури (α={abl['fixed_alpha']}, val)")
plt.show()
plots.plot_kd_ablation([a for a, _ in ap], [v for _, v in ap],
                       xlabel="α (вага CE)", baseline_val=abl["teacher_val"],
                       title=f"2-bit учень: свіп α (T={abl['fixed_temp']:.0f}, val)")
plt.show()
for T, v in tp: print(f"  T={T}: val {v*100:.2f}%")
for a, v in ap: print(f"  alpha={a}: val {v*100:.2f}%")
""")

md(f"""
**Що видно.** По **температурі** — у нашому діапазоні вища T стабільно краща: від
{pct(temp_pts[0][1])} при T={temp_pts[0][0]:.0f} до {pct(best_T[1])} при T={best_T[0]:.0f}
(монотонно). М'якший розподіл вчителя несе більше «темних знань», і 2-бітному учневі це
корисно; ще вищі T варто було б перевірити окремо. По **α** найкраще — {best_a[0]}
({pct(best_a[1])} val), і взагалі вага в бік дистиляції допомагає: {(
  'чиста дистиляція (α=0, ' + pct(a_pure_kd) + ') випереджає чистий CE (α=1, ' + pct(a_pure_ce) + '), '
  'а невелика домішка CE (α≈0.3) — трохи краща за обидві крайнощі.'
) if (a_pure_kd is not None and a_pure_ce is not None) else 'проміжні α працюють краще за крайні.'}
Тобто саме м'які таргети — корисний сигнал тут; жорсткі мітки радше страхують.
(Головний прогін використав T={cfg['T']:.0f}, α={cfg['alpha']} — розумний дефолт; свіп
показує, що T={best_T[0]:.0f}/α={best_a[0]} витиснули б ще трохи.)
""")

# --------------------------------------------------------------------------- #
if na:
    na_ce, na_kd = na["ce"]["test"], na["kd"]["test"]
    na_nce, na_nkd = na["netaug_ce"]["test"], na["netaug_kd"]["test"]
    kd_g, aug_g, both_g = na_kd - na_ce, na_nce - na_ce, na_nkd - na_ce
    aug_on_kd = na_nkd - na_kd
    ncfg = na["config"]
    fewer = na["full_params"] / na["base_params"]
    corners = [("CE", na_ce), ("KD", na_kd), ("NetAug+CE", na_nce), ("NetAug+KD", na_nkd)]
    best_corner = max(corners, key=lambda t: t[1])
    _bold = lambda v: f"**{pct(v)}**" if abs(v - best_corner[1]) < 1e-9 else pct(v)
    _sign = lambda g: ("помітно допомагає" if g > 0.007 else       # >0.7 pp
                       "трохи допомагає" if g > 0.002 else          # >0.2 pp
                       "майже не впливає" if g >= -0.002 else "радше шкодить")
    md(f"""
## 6. Бонус — NetAug (Network Augmentation) + KD

**Ідея.** На відміну від великих мереж, **маленькі моделі недонавчаються**, а не
переднавчаються — тож звична регуляризація (dropout, аугментація даних) їм радше
шкодить. NetAug робить навпаки: під час навчання **розширює ємність** мережі. Цільова
маленька модель — це під-мережа ширшої *augmented*-мережі зі **спільними вагами**; на
кожному кроці forward'имо і базу, і розширений варіант, а лоси сумуємо. Розширений
forward проштовхує додатковий градієнт крізь спільні ваги. На inference лишається
**тільки база**. Джерело: Cai et al., «Network Augmentation for Tiny Deep Learning»,
ICLR 2022 ([arXiv:2110.08890](https://arxiv.org/abs/2110.08890)).

**Наш сетап.** Ціль — **width-compressed VGG11 ({na['base_mult']}× канали)**,
{na['base_params']/1e6:.2f}M параметрів (у {fewer:.0f}× менше за вчителя) — справді мала
модель, той режим, де NetAug має сенс. Це четверта вісь стиснення (по ширині) на додачу
до прунінгу/кванту з ядра. Augmentation — до повної ширини (×{1/na['base_mult']:.0f}), що
відповідає паперу (помірний фактор). **Інтеграція з KD:** і база, і augmented-гілка
вчаться під м'які таргети того самого вчителя (`netaug_train` бере `DistillLoss`).
Конфіг: T={ncfg['T']:.0f}, α={ncfg['alpha']}, λ_aug={ncfg['aug_weight']:.0f},
base={na['base_mult']}×, aug={ncfg.get('aug_mult', 1.0)}×, {ncfg['epochs']} epochs,
seed={ncfg['seed']}.

**Чесне порівняння 2×2** — усе з того самого init/seed, {ncfg['epochs']} епох, однаковий
оптимізатор; різниця лише в методі:

|  | CE-лос | KD-лос |
|---|---|---|
| **звичайне навчання** | {_bold(na_ce)} | {_bold(na_kd)} |
| **NetAug** | {_bold(na_nce)} | {_bold(na_nkd)} |
""")
    code("""
na = json.load(open(f"{RESULTS}/netaug.json"))
cells = {k: na[k]["test"] for k in ["ce", "kd", "netaug_ce", "netaug_kd"]}
plots.plot_netaug_2x2(cells, teacher_acc=na["teacher_test"], full_acc=na["teacher_test"],
                      title=f"NetAug × KD на tiny VGG11 ({na['base_mult']}×, тест)")
plt.show()
print(f"tiny {na['base_mult']}x VGG11 ({na['base_params']/1e6:.2f}M), тест:")
for k in ["ce", "kd", "netaug_ce", "netaug_kd"]:
    print(f"  {k:<11}: {na[k]['test']*100:.2f}%")
print(f"KD {(na['kd']['test']-na['ce']['test'])*100:+.2f} | "
      f"NetAug {(na['netaug_ce']['test']-na['ce']['test'])*100:+.2f} | "
      f"NetAug+KD {(na['netaug_kd']['test']-na['ce']['test'])*100:+.2f} п.п. (vs CE)")
""")
    md(f"""
**Що видно (проти чистого CE).** KD {_sign(kd_g)} ({pp(kd_g)}), NetAug {_sign(aug_g)}
({pp(aug_g)}), разом NetAug+KD — {pp(both_g)} над CE. Найкращий кут — **{best_corner[0]}**
({pct(best_corner[1])}). {'NetAug дав ще ' + pp(aug_on_kd) + ' поверх KD.' if aug_on_kd > 0.003 else 'У нашому **одному запуску** NetAug не покращив KD (поверх KD ' + pp(aug_on_kd) + '), але за одного seed це цілком може бути шумом — для статистичного висновку потрібні 3–5 seed і mean±std. Правдоподібна гіпотеза: KD і NetAug дають той самий тип додаткового сигналу недонавченій моделі, тож коли вже є сильний вчитель, augmented-гілка докидає мало.'}
Обидві техніки — про **навчання**, не про розмір: на inference база однакова
({na['base_params']/1e6:.2f}M) незалежно від методу.
""")

# --------------------------------------------------------------------------- #
md(f"""
## 7. Як ми це робили: підходи та спроби

**Стартова точка.** Нічого не перенавчали з нуля: вчитель — готова fp32-VGG11 з ДЗ1,
стиснення — код прунінгу (ДЗ1) і квантизації/QAT (ДЗ2). Зверху дописали лише
дистиляцію (`src/distill.py`).

**Один код для чесного порівняння.** Щоб KD-vs-CE було чесним, зробили так, що
`alpha=1` у KD-лоссі точно дорівнює крос-ентропії — тоді CE-гілка й KD-гілка йдуть
**одним циклом** з тими самими оптимізатором/розкладом/сідом, і різниця лишається
тільки в лоссі. Для квант-учня це той самий centroid-QAT із ДЗ2, у який ми додали
опційного вчителя (одна гілка `if teacher is not None`).

**Технічна пастка з градієнтами.** Логіти вчителя рахуємо під `torch.no_grad()`, а не
`inference_mode`: inference-тензори не можна підмішувати в autograd-граф учня (вони
йдуть як константний таргет у KL). Через це спершу ловили помилку — і саме тому в коді
стоїть `no_grad`, а не `inference_mode`.

**Чого очікували й що вийшло.** Гіпотеза була проста: KD найкорисніший там, де
стиснення найсильніше. Дані її підтвердили — на 2 бітах виграш {pp(R[hard]['dtest'])},
а на прунінгу майже нуль. Це не «KD не працює», а «на прунінгу CE вже й так відновлює
модель до вчителя, стелі майже нема».

## 8. Звіт: висновки і рефлексія

**Що спрацювало добре.** KD-пайплайн (CE + T²·KL) стабільно й без зусиль інтегрувався і
з прунінгом, і з centroid-QAT. Виграш дистиляції у цьому запуску тим більший, чим сильніше
стиснення (тенденція single-seed): найбільший на найагресивнішому учні
({R[best_r]['tag']}, {R[best_r]['comp']:.1f}×,
{pp(R[best_r]['dtest'])} на тесті), і саме він — найкраща точка сумарно
({pct(R[best_r]['kd_test'])}). Self-distillation працює навіть за однакової архітектури
вчителя й учня. Ablation: вища температура (T={best_T[0]:.0f} найкраща у нашому діапазоні)
і перевага дистиляції в суміші (α={best_a[0]}; чистий CE α=1 — найгірший) — корисний
сигнал тут саме м'які таргети.

**Що вийшло не так, як очікувалось.** Виграш скромний — десяті частки п.п. На прунінг-учні
він майже нульовий ({pp(R['pruned']['dtest']) if 'pruned' in R else '—'}) — 80%-розріджена
модель зберігає fp32-точність ненульових ваг і відновлюється майже до вчителя і без KD,
тож дистиляції нема куди тягнути. Ефект KD впирається не в саму дистиляцію, а в те,
**скільки точності реально втрачено при стисненні** — там, де втрата мала, малий і виграш.

**Чому результати можуть бути не ідеальними.** (1) Вчитель — сам лише {pct(teacher['test'])},
тобто «стеля» дистиляції невисока; сильніший учитель (ширша мережа чи ансамбль) дав би
кращі м'які таргети. (2) Self-distillation з однаковою архітектурою обмежене: учень не
«слабший клас» моделей, а та сама мережа зі стисненими вагами, тож простір для
перенесення знань вужчий. (3) Гіперпараметри (T, α) підбирали на 2-бітному учні й
перенесли на решту — оптимум для кожного режиму міг би трохи відрізнятися. (4) Бюджет
епох невеликий; довше до-навчання трохи підняло б обидві гілки.

**Як покращити.** Сильніший/ширший учитель або ансамбль; feature-based дистиляція (не
лише логіти, а й проміжні активації — FitNets/attention transfer); підбір (T, α) під
кожен режим окремо; поєднати з бонусом **NetAug** (тренувати стиснуту мережу, тимчасово
розширюючи її ємність), щоб дати учневі більше «простору» під час навчання; для
Raspberry Pi з фінального проекту — дистиляція одночасно зі quantization-aware
експортом у INT8.
""")

md(f"""
## 9. Відтворення

```bash
python run_distill.py     --baseline ../hw1/results/baseline.pt --data-dir ./data --out results
python run_kd_ablation.py --baseline ../hw1/results/baseline.pt --data-dir ./data --out results
python run_netaug.py      --baseline ../hw1/results/baseline.pt --data-dir ./data --out results   # бонус
python build_notebook.py results
python -m pytest tests/          # KD-loss + NetAug elastic-VGG weight sharing
```

Код у `src/`: `distill.py` (KD-loss + KD-цикл навчання), `qat.py` (centroid-QAT з ДЗ2,
розширений опційним учителем), `netaug.py` (elastic-VGG11 + NetAug-цикл, бонус);
`model/data/engine/prune/kmeans_quant/plots/utils` — реюз із ДЗ1/ДЗ2. Версії пакетів —
у `requirements.txt`.
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
print("wrote", OUT_NB, "with", len(cells), "cells")

# --------------------------------------------------------------------------- #
def report_rows():
    rows = []
    for r in regimes:
        x = R[r]
        rows.append(f"| {x['tag']} | {x['comp']:.1f}× | {pct(x['ce_test'])} | "
                    f"{pct(x['kd_test'])} | {pp(x['dtest'])} |")
    return "\n".join(rows)


if na:
    _na_ce, _na_kd = na["ce"]["test"], na["kd"]["test"]
    _na_nce, _na_nkd = na["netaug_ce"]["test"], na["netaug_kd"]["test"]
    _kd_g, _aug_g, _both_g = _na_kd - _na_ce, _na_nce - _na_ce, _na_nkd - _na_ce
    _aug_on_kd = _na_nkd - _na_kd
    _best = max([("CE", _na_ce), ("KD", _na_kd), ("NetAug+CE", _na_nce),
                 ("NetAug+KD", _na_nkd)], key=lambda t: t[1])
    _fewer = na["full_params"] / na["base_params"]
    _nc = na["config"]
    _bold_r = lambda v: f"**{pct(v)}**" if abs(v - _best[1]) < 1e-9 else pct(v)
    bonus_report = f"""
## Бонус — NetAug + KD (tiny {na['base_mult']}× VGG11)
NetAug виходить з того, що **малі моделі недонавчаються**: замість регуляризації він під час
навчання розширює ємність — цільова мала модель є під-мережею ширшої augmented-мережі зі
спільними вагами; forward'имо обидві, лоси сумуємо, на inference лишається лише база.
Джерело: Cai et al., «Network Augmentation for Tiny Deep Learning», ICLR 2022
([arXiv:2110.08890](https://arxiv.org/abs/2110.08890)). Ціль — width-compressed VGG11
({na['base_mult']}×, {na['base_params']/1e6:.2f}M, у {_fewer:.0f}× менше за вчителя).
Інтегрували з KD: обидві гілки вчаться під м'які таргети вчителя. Чесне 2×2 (той самий
init/seed). **Конфіг:** T={_nc['T']:.0f}, α={_nc['alpha']}, λ_aug={_nc['aug_weight']:.0f},
base={na['base_mult']}×, aug={_nc.get('aug_mult', 1.0)}×, {_nc['epochs']} epochs,
seed={_nc['seed']}.

| tiny {na['base_mult']}× VGG11 | CE-лос | KD-лос |
|---|---|---|
| звичайне навчання | {_bold_r(_na_ce)} | {_bold_r(_na_kd)} |
| NetAug | {_bold_r(_na_nce)} | {_bold_r(_na_nkd)} |

Проти чистого CE: KD {pp(_kd_g)}, NetAug {pp(_aug_g)}, разом {pp(_both_g)}; найкращий кут —
**{_best[0]}** ({pct(_best[1])}). У нашому **одному запуску** NetAug не покращив KD (поверх KD
{pp(_aug_on_kd)}) — але за одного seed різниця такого масштабу цілком може бути шумом; для
статистичного висновку потрібні 3–5 seed і mean±std. Правдоподібна гіпотеза: KD і NetAug
дають той самий тип додаткового сигналу недонавченій моделі, тож коли вже є сильний
teacher-сигнал, augmented-гілка докидає мало (і ймовірно, при 4× augmentation спільні ваги
тягне радше до повної ширини, ніж до бази). Обидві техніки — про навчання, не про розмір:
база на inference однакова ({na['base_params']/1e6:.2f}M).
"""
else:
    bonus_report = ""


report = f"""# Звіт — ДЗ 3: Self-Distillation зі стиснутими моделями (VGG11 / CIFAR-10)

Код і результати в репо: [imagic9/efficient-ml-set → hw3](https://github.com/imagic9/efficient-ml-set/tree/main/hw3)

## Методологія
Вчитель — непорушена fp32-VGG11 з ДЗ1 ({pct(teacher['test'])} на тесті), заморожена, не
перенавчається. Кожен стиснутий учень до-навчається **двічі з однакового старту й seed**:
CE-only (baseline) і KD — однаковим кодом, різниця лише в лоссі, тож виграш чисто за
дистиляцією. Усі проміжні числа (криві, ablation-свіпи) — на валідації; тест — рівно
один раз на фінальну модель.

## KD-loss
$L = \\alpha\\,CE + (1-\\alpha)\\,T^2\\,KL(\\text{{teacher}}_T \\| \\text{{student}}_T)$.
`alpha=1` збігається з крос-ентропією (спільний код для CE- і KD-гілки); множник $T^2$
тримає масштаб градієнта м'якої частини.

## Підсумкова таблиця (тест — раз на модель, T={cfg['T']:.0f}, α={cfg['alpha']})

| Стиснутий учень | Стиснення | CE-only | KD | KD − CE |
|---|---|---|---|---|
| Вчитель fp32 | 1× | — | {pct(teacher['test'])} | — |
{report_rows()}
{bonus_report}
## Як ми робили (підходи та спроби)
Нічого не перенавчали: вчитель — fp32-модель з ДЗ1, стиснення — код прунінгу (ДЗ1) і
квантизації/QAT (ДЗ2), зверху лише дистиляція. Щоб KD-vs-CE було чесним, зробили
`alpha=1` точно рівним крос-ентропії — тоді обидві гілки йдуть одним циклом з тими самими
оптимізатором/розкладом/сідом. Для квант-учня додали опційного вчителя прямо в
centroid-QAT із ДЗ2. Технічна пастка: логіти вчителя рахуємо під `torch.no_grad()` (не
`inference_mode`), бо inference-тензори не можна підмішувати в autograd-граф учня —
на цьому спершу спіткнулися. Гіпотезу «KD найкорисніший там, де стиснення найсильніше»
дані підтвердили.

## What worked well
KD-пайплайн стабільно інтегрувався і з прунінгом, і з QAT. Виграш дистиляції у цьому
запуску тим більший, чим сильніше стиснення (тенденція single-seed, не строгий закон):
найбільший на найагресивнішому учні ({R[best_r]['tag']}, {R[best_r]['comp']:.1f}×,
{pp(R[best_r]['dtest'])} на тесті), і саме він — найкраща точка сумарно
({pct(R[best_r]['kd_test'])}). Ablation: вища температура (T={best_T[0]:.0f} найкраща) і
перевага дистиляції в суміші (α={best_a[0]}; чистий CE α=1 — найгірший) — тобто корисний
сигнал тут саме м'які таргети.

## What didn't turn out as expected
Виграш скромний — десяті частки п.п. На прунінг-учні він майже нульовий
({pp(R['pruned']['dtest']) if 'pruned' in R else '—'}): 80%-розріджена модель зберігає
fp32-точність ненульових ваг і без KD відновлюється майже до вчителя. Тобто ефект KD
впирається в те, **скільки точності реально втрачено при стисненні**, а не в саму
дистиляцію.

## Why results might not be great
Вчитель сам лише {pct(teacher['test'])} — «стеля» дистиляції невисока (сильніший/ширший
учитель дав би кращі м'які таргети). Self-distillation з однаковою архітектурою обмежене:
учень — не слабший клас моделей, а та сама мережа зі стисненими вагами. Гіперпараметри
(T, α) підбирали на 2-бітному учні й перенесли на решту. Бюджет епох невеликий.

## How to improve
Сильніший/ширший учитель або ансамбль; feature-based дистиляція (проміжні активації, не
лише логіти); підбір (T, α) під кожен режим; поєднати дистиляцію зі стисненням в одному
циклі (KD прямо під час QAT/прунінгу кожного учня, а не лише як окреме відновлення);
{'NetAug (бонус нижче) на ще менших моделях або з меншим фактором augmentation; ' if na else ''}для
Raspberry Pi з фінального проекту — дистиляція разом із QAT-експортом у INT8.
"""
with open(OUT_REPORT, "w") as f:
    f.write(report)
print("wrote", OUT_REPORT)
