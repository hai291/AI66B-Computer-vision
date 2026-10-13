Issue #4 - CNN baseline, ResNet tu xay va ResNet thu vien (song song voi DenseNet issue #5)

Bo cuc folder (chay moi lenh tu thu muc repo):
  resnet/model_resnet_manual.py   - ResNet-18 tu viet tay (BasicBlock + ResNet)
  resnet/model_resnet.py          - ResNet-18 thu vien torchvision (adapt 1 kenh, 3 lop)
  resnet/train_resnet.py          - train/evaluate chung cho baseline + 2 ban ResNet
  resnet/test_resnet_pipeline.py  - smoke test: DICOM gia + train 1 epoch ca 3 model
  resnet/summarize_resnet_results.py - tong hop + ve bieu do tu CSV da luu
  resnet/resnet_manual_kaggle.ipynb  - notebook Kaggle: chi train ban tu viet tay
  resnet/resnet_kaggle.ipynb         - notebook Kaggle: chi train ban thu vien
  resnet/compare_resnet_kaggle.ipynb - notebook Kaggle: train ca 3 + so sanh test
  resnet/README.md                - file nay (tai lieu cach lam + lenh chay)

Yeu cau duoc thuc hien bang ba model, tai su dung pipeline cua train_densenet.py:
1. baseline: CNN co ban 3 cum Conv/BatchNorm/ReLU/MaxPool (model_baseline.py o thu muc repo, giu nguyen).
2. resnet_manual: tu dung ResNet-18 tu cac lop torch.nn, khong goi model co san.
3. resnet: goi torchvision.models.resnet18.

Ban tu dung nam trong model_resnet_manual.py: lop BasicBlock (y = ReLU(F(x) + shortcut(x)))
va lop ResNet dung stem conv 7x7 + 4 stage [2, 2, 2, 2] BasicBlock, khong dung torchvision
hoac trong so tai san. Ban thu vien nam trong model_resnet.py, adapt conv1 ve 1 kenh xam
va fc ve 3 lop; khi --pretrained thi cong tong trong so RGB sang 1 kenh (giong DenseNet).

ResNet giu dac trung cu bang skip connection cong truc tiep, hoc them phan du F(x).
Vi du: vao 64 kenh -> 2 conv 3x3 giu 64 kenh -> cong dau vao; khac shape thi dung conv 1x1
projection (stride=2 khi giam khong gian + tang kenh: 64 -> 128 -> 256 -> 512).
ResNet-18 mot kenh, 3 dau ra: 11.171.779 tham so; ban tu dung va ban thu vien bang nhau
(41 lop Conv/BatchNorm/Linear, da kiem chung copy trong so cho cung output).

"Khong goi thu vien" o day la khong goi kien truc model dung san.
Code van dung PyTorch cho Conv, BatchNorm, pooling, gradient va optimizer.
Nguon kien truc: https://arxiv.org/abs/1512.03385
Doi chieu cau truc: https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html

Chia du lieu (giong het issue #5)
Train / validation / test xap xi 70% / 15% / 15%, seed 42, theo study_id.
So luong duoc lam tron theo so ca; anh cung ca chi thuoc mot tap.
Du lieu test duoc giu lai tu train.csv co nhan, khong phai test an cua Kaggle.
split.csv luu moi ca thuoc tap nao. Ba model dung chung file nay.
Giu nguyen du lieu va MAX_STUDIES; khong tao split moi cho tung model.

Train dung de cap nhat trong so.
Validation danh gia sau moi epoch va chon checkpoint co log loss thap nhat.
Test chi evaluate sau khi chot cau hinh va train xong; khong dung test de chon model/epoch.
Ca ba model duoc danh gia tren cung crop test voi accuracy, macro-F1 va log loss.
Checkpoint cu chia 80/20 khong duoc dung lam ket qua moi: phai train lai.

Xu ly anh chung (copy tu train_densenet.py)
Ghep toa do va nhan bang study_id + condition + level, bo nhan thieu.
Crop 128x128 quanh toa do, them vien khi sat mep, anh xam min-max ve [0, 1].
Mac dinh anh dau vao 128x128, batch 16, Adam lr=0.001, khong augmentation.
Ca ba mac dinh hoc tu dau, khong pretrained de so sanh cung dieu kien.
Co the chay rieng mot thu nghiem thu vien voi --pretrained;
khi bao cao phai ghi ro co them kien thuc tu ImageNet, khong gop voi so sanh tu dau.
Cung ngan sach epoch khong dam bao moi kien truc da dat cau hinh tot nhat.

Lenh chay tu thu muc repo (file rieng, khong dung chung output voi DenseNet)
python resnet/train_resnet.py --data-root data --model baseline --split-file runs/resnet_full/split.csv --output runs/resnet_full/baseline
python resnet/train_resnet.py --data-root data --model resnet_manual --split-file runs/resnet_full/split.csv --output runs/resnet_full/resnet_manual
python resnet/train_resnet.py --data-root data --model resnet --split-file runs/resnet_full/split.csv --output runs/resnet_full/resnet

Sau khi train xong va da chot cau hinh:
python resnet/train_resnet.py --data-root data --evaluate runs/resnet_full/baseline
python resnet/train_resnet.py --data-root data --evaluate runs/resnet_full/resnet_manual
python resnet/train_resnet.py --data-root data --evaluate runs/resnet_full/resnet
Them --split val neu chi muon kiem tra lai validation; mac dinh --evaluate dung test.
--max-studies 20 --epochs 1 de chay thu; dung split-file/output rieng cho smoke test.

Chay tren Kaggle (khong can clone GitHub, moi notebook tu chua toan bo code)
  resnet/resnet_manual_kaggle.ipynb   - chi train ban tu viet tay.
  resnet/resnet_kaggle.ipynb          - chi train ban thu vien.
  resnet/compare_resnet_kaggle.ipynb  - train tung model rieng, cell cuoi moi evaluate
    test va tao comparison.csv (khuyen dung cho bao cao).
Cach chay: tai notebook len Kaggle, gan dataset RSNA o Add Input, bat GPU (T4),
chay lan luot cac cell. Mac dinh chay thu 20 ca, 1 epoch de kiem tra pipeline.
De lay ket qua day du: doi MAX_STUDIES=0, EPOCHS=5, RUN_ROOT sang thu muc moi
roi chay lai tu cell cau hinh. Tinh chinh dua tren validation truoc khi chay cell test.
Muon dung lai split cua lan chay dau: tai split.csv ra, gan vao notebook khac
roi sua SPLIT_FILE. Khong so sanh model chay 20 ca voi model dung toan bo ca.

Dau ra moi model
config.json va samples.csv: cau hinh, nhan crop va tap tuong ung.
history.csv: train loss va validation metric theo epoch.
best.pt: checkpoint duoc chon bang validation.
val_metrics.json, val_predictions.csv: ket qua validation tot nhat.
test_metrics.json, test_predictions.csv: chi xuat khi evaluate test.
Confusion matrix co hang la nhan that, cot la du doan; thu tu Normal/Mild, Moderate, Severe.
comparison.csv trong notebook tong hop metric test cua cac model da chay.

Kiem tra code tren may (chay tu thu muc repo)
python resnet/test_resnet_pipeline.py
Test tao DICOM gia, kiem tra chia ca khong trung, chay train/luu/nap/evaluate ca ba.
Anh test duoc co y lam hong trong luc train de xac nhan vong train khong doc chung.
Test doi chieu ket qua so cua ResNet tu xay va thu vien khi gan cung trong so
(11.171.779 params, 41 lop, numerical agreement rtol=1e-5).
Viec gan trong so tham chieu nay chi nam trong test; model tu xay that hoc tu dau.

Tong hop va ve lai tu cac CSV da luu (khong can GPU, MRI hay checkpoint):
python resnet/summarize_resnet_results.py results/issue4_resnet_smoke_v1
Lenh kiem tra split, cau hinh va metric roi tao comparison.csv, split_summary.csv,
test_comparison.png va test_confusion_matrices.png ngay trong thu muc ket qua.

Uu diem: ResNet tai su dung dac trung bang skip connection; baseline nho va de giai thich.
Han che: chua do thoi gian/bo nho de so sanh thuc te.
Metric tinh theo crop voi toa do da co, chua danh gia kha nang tu tim vung can crop.
Crop co dinh theo pixel co the mat ngu canh va khac kich thuoc vat ly giua may chup.
Khong dung diem nay nhu diem RSNA day du. PR van can thanh vien khac review truoc merge.
