# hybrid-fuzzy-ensemble
for running the venv in projext dir use:

```bash
.venv\Scripts\activate
```

libs already installed in requirements.txt after adding new libs pls use :

```bash
pip freeze > requirements.txt
```

file structure :

hybrid-fuzzy-ensemble
│
├── data
│   ├── raw
│   └── processed
│
├── src
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── fuzzy.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── results
│   └── metrics.csv   (بعداً ساخته می‌شود)
│
├── run.py
├── requirements.txt
└── README.md


dataset link and info in :
https://archive.ics.uci.edu/dataset/45/heart+disease


هدف

ایجاد یک شبکه عصبی ترکیبی (Hybrid Ensemble) برای پیش‌بینی Heart Disease که شامل:

دو شبکه عصبی پایه (BaseNN1 و BaseNN2)

لایه فازی Voting برای ترکیب خروجی‌ها با وزن‌دهی بر اساس اعتماد (confidence)

شبکه نهایی (Meta-Network) که خروجی فازی را پردازش می‌کند

هدف اصلی: افزایش دقت و پایداری مدل نسبت به استفاده از یک شبکه تنها.

2. ساختار مدل
2.1 BaseNN1

عمق کم: 2 لایه مخفی

Activation: ReLU

Dropout: 0.2

خروجی: Sigmoid (برای احتمال حضور بیماری)

2.2 BaseNN2

عمق بیشتر: 3 لایه مخفی

Activation: Tanh

Dropout: 0.1

خروجی: Sigmoid

2.3 Voting Layer (Fuzzy)

ترکیب خروجی دو شبکه

وزن‌دهی بر اساس confidence: هر خروجی نزدیک به 0 یا 1 وزن بیشتری دارد

استفاده از قواعد ساده فازی:

اگر هر دو شبکه نزدیک به 0 یا 1 باشند، نتیجه نهایی همان باشد

اگر شبکه‌ها اختلاف داشته باشند، میانگین فازی گرفته شود

2.4 Meta Network (Final Layer)

یک لایه Linear ساده برای خروجی نهایی

Activation: Sigmoid

3. مزایا

استفاده از دو شبکه متفاوت باعث می‌شود هر شبکه بخش‌های متفاوتی از ویژگی‌ها را یاد بگیرد

Ensemble کاهش خطای واریانس و overfitting را فراهم می‌کند

Voting Layer فازی باعث تصمیم‌گیری هوشمندتر و مقاوم‌تر می‌شود




این دیتاست از UCI Machine Learning Repository گرفته شده است و هدف آن تشخیص حضور یا عدم حضور بیماری قلبی است.
UCI Machine Learning Repository

2. تعداد رکوردها و ویژگی‌ها

شامل 303 رکورد از بیماران است.
UCI Machine Learning Repository

دارای 76 ویژگی در فایل اصلی، اما مدل‌های پژوهشی معمولاً از 14 ویژگی اصلی استفاده می‌کنند.
UCI Machine Learning Repository

3. متغیر هدف (Target)

مقدار هدف در دیتاست به شکل عدد از 0 تا 4 است که:

0 = عدم بیماری (Healthy)

1–4 = نشانه‌های مختلف بیماری قلبی

معمولاً برای مساله‌ی طبقه‌بندی دودویی (Binary Classification)، مقادیر 1 تا 4 به 1 تبدیل می‌شوند یعنی وجود بیماری؛ و 0 همان عدم وجود بیماری.
archive-beta.ics.uci.edu

4. برخی از ویژگی‌های مهم

دیتاست شامل ویژگی‌های پزشکی کلی مثل سن، فشار خون، کلسترول، نوع درد قفسه سینه و … است.
IBM Cloud Pak for Data

برخی ویژگی‌های معمول (از مجموعه 14 ویژگی انتخاب‌شده):

age: سن فرد

sex: جنسیت (۱=مرد، ۰=زن)

cp: نوع درد قفسه‌ی سینه

trestbps: فشار خون در حالت استراحت

chol: کلسترول سرم

fbs: قند خون ناشتا

restecg: علائم ECG استراحت

thalach: حداکثر ضربان قلب

exang: آنژین القا شده با ورزش

oldpeak: ST depression induced by exercise

slope, ca, thal: ویژگی‌های دیگر مربوط به ECG و مقادیر تست‌های پزشکی

num: متغیر هدف (0 تا 4)
IBM Cloud Pak for Data



(.venv) C:\Users\ME\Desktop\hybrid-fuzzy-ensemble>python run.py
==== Training Hybrid Fuzzy Ensemble ====
Epoch [1/50] - Train Loss: 0.7000, Val Loss: 0.6800, Val Acc: 0.4590
Epoch [2/50] - Train Loss: 0.6426, Val Loss: 0.6125, Val Acc: 0.7705
Epoch [3/50] - Train Loss: 0.5837, Val Loss: 0.5824, Val Acc: 0.8033
Epoch [4/50] - Train Loss: 0.5606, Val Loss: 0.5659, Val Acc: 0.8197
Epoch [5/50] - Train Loss: 0.5463, Val Loss: 0.5543, Val Acc: 0.8525
Epoch [6/50] - Train Loss: 0.5366, Val Loss: 0.5461, Val Acc: 0.8525
Epoch [7/50] - Train Loss: 0.5288, Val Loss: 0.5387, Val Acc: 0.8525
Epoch [8/50] - Train Loss: 0.5221, Val Loss: 0.5350, Val Acc: 0.8525
Epoch [9/50] - Train Loss: 0.5165, Val Loss: 0.5289, Val Acc: 0.8525
Epoch [10/50] - Train Loss: 0.5105, Val Loss: 0.5263, Val Acc: 0.8525
Epoch [11/50] - Train Loss: 0.5073, Val Loss: 0.5195, Val Acc: 0.8525
Epoch [12/50] - Train Loss: 0.5038, Val Loss: 0.5215, Val Acc: 0.8525
Epoch [13/50] - Train Loss: 0.4975, Val Loss: 0.5202, Val Acc: 0.8525
Epoch [14/50] - Train Loss: 0.4925, Val Loss: 0.5187, Val Acc: 0.8361
Epoch [15/50] - Train Loss: 0.4894, Val Loss: 0.5109, Val Acc: 0.8525
Epoch [16/50] - Train Loss: 0.4849, Val Loss: 0.5149, Val Acc: 0.8361
Epoch [17/50] - Train Loss: 0.4811, Val Loss: 0.5106, Val Acc: 0.8361
Epoch [18/50] - Train Loss: 0.4781, Val Loss: 0.5098, Val Acc: 0.8361
Epoch [19/50] - Train Loss: 0.4743, Val Loss: 0.5062, Val Acc: 0.8525
Epoch [20/50] - Train Loss: 0.4721, Val Loss: 0.5047, Val Acc: 0.8525
Epoch [21/50] - Train Loss: 0.4692, Val Loss: 0.5047, Val Acc: 0.8525
Epoch [22/50] - Train Loss: 0.4659, Val Loss: 0.5052, Val Acc: 0.8361
Epoch [23/50] - Train Loss: 0.4633, Val Loss: 0.4999, Val Acc: 0.8525
Epoch [24/50] - Train Loss: 0.4610, Val Loss: 0.5024, Val Acc: 0.8361
Epoch [25/50] - Train Loss: 0.4591, Val Loss: 0.4985, Val Acc: 0.8525
Epoch [26/50] - Train Loss: 0.4565, Val Loss: 0.4965, Val Acc: 0.8525
Epoch [27/50] - Train Loss: 0.4539, Val Loss: 0.4949, Val Acc: 0.8525
Epoch [28/50] - Train Loss: 0.4524, Val Loss: 0.4954, Val Acc: 0.8361
Epoch [29/50] - Train Loss: 0.4505, Val Loss: 0.4953, Val Acc: 0.8361
Epoch [30/50] - Train Loss: 0.4484, Val Loss: 0.4936, Val Acc: 0.8361
Epoch [31/50] - Train Loss: 0.4492, Val Loss: 0.4895, Val Acc: 0.8525
Epoch [32/50] - Train Loss: 0.4448, Val Loss: 0.4930, Val Acc: 0.8361
Epoch [33/50] - Train Loss: 0.4428, Val Loss: 0.4874, Val Acc: 0.8525
Epoch [34/50] - Train Loss: 0.4409, Val Loss: 0.4878, Val Acc: 0.8361
Epoch [35/50] - Train Loss: 0.4391, Val Loss: 0.4873, Val Acc: 0.8361
Epoch [36/50] - Train Loss: 0.4377, Val Loss: 0.4876, Val Acc: 0.8361
Epoch [37/50] - Train Loss: 0.4363, Val Loss: 0.4867, Val Acc: 0.8361
Epoch [38/50] - Train Loss: 0.4345, Val Loss: 0.4846, Val Acc: 0.8525
Epoch [39/50] - Train Loss: 0.4327, Val Loss: 0.4831, Val Acc: 0.8525
Epoch [40/50] - Train Loss: 0.4309, Val Loss: 0.4825, Val Acc: 0.8525
Epoch [41/50] - Train Loss: 0.4293, Val Loss: 0.4825, Val Acc: 0.8525
Epoch [42/50] - Train Loss: 0.4282, Val Loss: 0.4812, Val Acc: 0.8525
Epoch [43/50] - Train Loss: 0.4264, Val Loss: 0.4800, Val Acc: 0.8525
Epoch [44/50] - Train Loss: 0.4252, Val Loss: 0.4785, Val Acc: 0.8525
Epoch [45/50] - Train Loss: 0.4238, Val Loss: 0.4785, Val Acc: 0.8525
Epoch [46/50] - Train Loss: 0.4223, Val Loss: 0.4764, Val Acc: 0.8525
Epoch [47/50] - Train Loss: 0.4208, Val Loss: 0.4769, Val Acc: 0.8525
Epoch [48/50] - Train Loss: 0.4189, Val Loss: 0.4762, Val Acc: 0.8525
Epoch [49/50] - Train Loss: 0.4177, Val Loss: 0.4726, Val Acc: 0.8525
Epoch [50/50] - Train Loss: 0.4165, Val Loss: 0.4735, Val Acc: 0.8525
Training complete. Best model saved at: results/hybrid_model.pth

==== Evaluating Model ====
Confusion Matrix:
[[20  8]
 [ 1 32]]

Classification Report:
              precision    recall  f1-score   support

         0.0     0.9524    0.7143    0.8163        28
         1.0     0.8000    0.9697    0.8767        33

    accuracy                         0.8525        61
   macro avg     0.8762    0.8420    0.8465        61
weighted avg     0.8699    0.8525    0.8490        61


==== Visualizing Sample Predictions ====
True labels: [0. 0. 0. 0. 0. 0. 1. 0.]

==== Visualizing Model Graph ====
Graph saved as 'hybrid_ensemble_model_graph.png'

==== Visualizing Real Validation Samples with Confidence ====
True labels: [0. 0. 0. 0. 0.]
Predicted labels: [0. 0. 0. 1. 0.]
Confidence: [0.35335106 0.33181542 0.35823363 0.5515528  0.3547606 ]

(.venv) C:\Users\ME\Desktop\hybrid-fuzzy-ensemble>





معماری پیشنهادی (Proposed Architecture)

مدل پیشنهادی از سه بخش اصلی تشکیل شده است:

3.1 شبکه اول (BaseNN1 – شبکه کم‌عمق)

این شبکه به‌گونه‌ای طراحی شده که الگوهای ساده و خطی‌تر را از داده یاد بگیرد.

ویژگی‌ها:

2 لایه مخفی

تابع فعال‌سازی: ReLU

Dropout = 0.2

خروجی: Sigmoid (احتمال وجود بیماری)

هدف این شبکه:

یادگیری الگوهای پایه و عمومی بدون پیچیدگی زیاد

3.2 شبکه دوم (BaseNN2 – شبکه عمیق‌تر)

این شبکه نسبت به BaseNN1 عمیق‌تر است و با تابع فعال‌سازی متفاوت کار می‌کند تا فضای جستجوی متفاوتی را پوشش دهد.

ویژگی‌ها:

3 لایه مخفی

تابع فعال‌سازی: Tanh

Dropout = 0.1

خروجی: Sigmoid

هدف این شبکه:

یادگیری روابط غیرخطی‌تر و پیچیده‌تر بین ویژگی‌ها

استفاده از Tanh باعث می‌شود رفتار این شبکه از نظر توزیع گرادیان و پاسخ متفاوت از BaseNN1 باشد، که برای ensemble بسیار مهم است.

3.3 لایه رأی‌گیری فازی (Fuzzy Voting Layer)

نوآوری اصلی مدل در این بخش قرار دارد.

ایده اصلی:

به‌جای میانگین‌گیری ساده از خروجی شبکه‌ها، از یک وزن‌دهی مبتنی بر confidence استفاده شده است.

تعریف confidence:

اگر خروجی یک شبکه به 0 یا 1 نزدیک باشد → اعتماد بیشتر
اگر خروجی نزدیک به 0.5 باشد → عدم قطعیت بیشتر

فرمول استفاده‌شده:

confidence = |output - 0.5| × 2

نحوه ترکیب:
Weighted Output = (x1 × conf1 + x2 × conf2) / (conf1 + conf2)


نتیجه:

اگر هر دو شبکه مطمئن باشند → تصمیم قوی

اگر یکی مطمئن‌تر باشد → رأی آن غالب می‌شود

اگر هر دو نامطمئن باشند → تصمیم نرم و محتاطانه

این رفتار دقیقاً مشابه تصمیم‌گیری فازی انسانی است.

3.4 شبکه نهایی (Meta Network)

خروجی لایه فازی وارد یک لایه خطی ساده می‌شود:

Linear(1 → 1)

Sigmoid

هدف:

تنظیم نهایی خروجی ensemble و افزایش انعطاف‌پذیری تصمیم نهایی

4. فرآیند آموزش و ارزیابی

Optimizer: Adam

Loss Function: Binary Cross Entropy

Epochs: 50

Batch Size: 8

مدل روی داده‌های آموزش train شده و روی validation ارزیابی شده است.

5. نتایج تجربی (Experimental Results)

نتایج نشان می‌دهد که مدل:

Accuracy ≈ 85%

Recall برای کلاس بیمار ≈ 97%

خطای False Negative بسیار پایین

Confusion Matrix نمونه:

[[20  8]
 [ 1 32]]


این موضوع در کاربردهای پزشکی اهمیت زیادی دارد، زیرا از دست دادن بیمار (False Negative) هزینه بالایی دارد.

6. تحلیل و جمع‌بندی

مزایای مدل پیشنهادی:

کاهش واریانس نسبت به شبکه تکی

تصمیم‌گیری پایدارتر

استفاده از confidence در فرآیند ensemble

مناسب برای دیتاست‌های پزشکی کوچک

مدل پیشنهادی یک چارچوب منعطف است و می‌تواند به‌راحتی:

به شبکه‌های بیشتر گسترش یابد

با قوانین فازی پیچیده‌تر ترکیب شود

برای مسائل پزشکی دیگر استفاده شود
