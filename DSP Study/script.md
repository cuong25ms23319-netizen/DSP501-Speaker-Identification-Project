# Noi dung thuyet trinh — Speaker Identification System
# Match voi 25 slides trong Speaker_Identification_Slides.pptx

---

## Slide 1: Title

> Xin chao thay va cac ban. Hom nay nhom em se trinh bay ve de tai "Speaker Identification System" — he thong nhan dien nguoi noi su dung xu ly tin hieu so. Nhom gom 4 thanh vien: Cuong, Dan, Quang va Khoa, duoi su huong dan cua thay Dang Ngoc Minh Duc.

---

## Slide 2: Outline

> Bai trinh bay gom 9 phan chinh. Dau tien la gioi thieu van de va dong luc nghien cuu. Tiep theo la du lieu — 5 nguoi noi, 125 mau. Phan 3 la phuong phap DSP gom bo loc FIR va pre-emphasis. Phan 4 la trich xuat dac trung — so sanh Pipeline A va Pipeline B. Phan 5 la mo hinh SVM. Phan 6 la ket qua thuc nghiem voi cac bieu do tu notebook. Phan 7 la thao luan tai sao Pipeline B thang. Phan 8 la han che va ket luan. Va cuoi cung la demo truc tiep.

---

## Slide 3: Problem & Motivation

> Van de dat ra la: lam sao de may tinh nhan dien duoc "ai dang noi" chi tu mot doan audio ngan khoang 3 giay?
>
> Thach thuc lon nhat la tieng noi la tin hieu non-stationary — no thay doi lien tuc theo thoi gian. Ngoai ra, nhieu moi truong lam sai lech tin hieu, va mic khac nhau cho dac tinh khac nhau.
>
> Cau hoi nghien cuu cua nhom la: Viec tien xu ly bang DSP co cai thien do chinh xac nhan dien nguoi noi so voi cach lam don gian khong?
>
> De tra loi, nhom xay dung 2 pipeline. Pipeline A la baseline — lay tin hieu tho, trich 6 dac trung thoi gian, roi dua vao SVM. Pipeline B la DSP-enhanced — ap dung bo loc FIR, pre-emphasis, roi trich xuat 26 dac trung MFCC, cung dua vao SVM. Ca hai dung cung mot classifier de so sanh cong bang.

---

## Slide 4: Dataset

> Du lieu gom 5 nguoi noi: Dan, Cuong, Quang, Anne va Khoa. Moi nguoi thu 25 file audio, moi file dai khoang 3 giay. Tong cong la 125 mau.
>
> Tat ca file deu la mono WAV, tan so lay mau 16 kHz, tuong duong 48 ngan mau moi clip.
>
> Truoc khi phan tich, moi file duoc tien xu ly qua 4 buoc: buoc 1 la load audio ve mono 16 kHz. Buoc 2 la normalize bien do ve khoang -1 den 1 de cac file co cung muc am luong. Buoc 3 la cat bo khoang lang dau cuoi voi nguong -20 dB. Va buoc 4 la pad hoac crop ve dung 48000 mau de dong nhat kich thuoc dau vao cho SVM.

---

## Slide 5: Signal Analysis — Waveform

> Day la bieu do dang song cua 1 file mau truoc va sau khi loc. Hinh tren la tin hieu tho — raw signal. Hinh duoi la sau khi ap dung bo loc FIR bandpass 300 den 3400 Hz.
>
> Cac ban co the thay hinh dang tong the tuong tu nhau — vi phan lon nang luong giong noi nam trong dai tan 300 den 3400 Hz. Nhung tin hieu sau loc muot hon, phan im lang phang hon vi nhieu nen da bi loai bo.

---

## Slide 6: FIR Bandpass Filter (300–3400 Hz)

> Day la phan tich bo loc FIR ma nhom thiet ke.
>
> Bieu do ben trai la dap ung tan so — magnitude response. Dai thong 300 den 3400 Hz o muc 0 dB, nghia la giu nguyen tin hieu. Ngoai dai thi suy giam xuong duoi -40 dB, nghia la loai bo hon 99% nang luong nhieu.
>
> Bieu do giua la dap ung pha — phase response. Pha tuyen tinh trong dai thong, nghia la tin hieu khong bi meo dang khi di qua bo loc. Dieu nay quan trong cho MFCC vi FFT nhay cam voi pha.
>
> Bieu do ben phai la dap ung xung — chinh la cac he so bo loc. No doi xung qua diem giua, xac nhan day la bo loc FIR Type I voi pha tuyen tinh. Bo loc co 101 tap, su dung cua so Hamming.

---

## Slide 7: Spectrum Comparison (FFT)

> Day la so sanh pho tan so bang FFT. Duong xanh la tin hieu tho, duong cam la sau khi loc.
>
> Trong dai 300 den 3400 Hz, hai duong gan trung nhau — bo loc giu nguyen thong tin giong noi. Nhung ngoai dai, duong cam suy giam manh — nhieu da bi loai bo.
>
> Hai duong doc mau do va xanh la la ranh gioi cua bo loc. Phan bi cat chu yeu la nhieu tan so thap nhu tieng hum dien va nhieu tan so cao — khong chua thong tin giong noi quan trong.

---

## Slide 8: Spectrogram (STFT) — Before vs After FIR

> Spectrogram cho thay tan so nao xuat hien tai thoi diem nao. Truc X la thoi gian, truc Y la tan so, mau sac the hien nang luong — vang sang la nang luong cao, tim toi la nang luong thap.
>
> Hinh ben trai la raw — nang luong trai deu tu 0 den 8000 Hz. Hinh ben phai la sau loc — nang luong chi tap trung trong dai 300 den 3400 Hz. Vung ngoai dai toi han.
>
> Cac vach ngang sang — chinh la formants, dac trung giong noi cua moi nguoi — van ro rang trong dai thong. Day la thong tin ma MFCC se trich xuat.

---

## Slide 9: Power Spectral Density (PSD)

> Day la mat do pho cong suat tinh bang phuong phap Welch — on dinh hon FFT vi lay trung binh nhieu doan.
>
> Duong xanh la raw, duong cam la sau loc. Trong dai 300 den 3400 Hz, hai duong gan trung nhau — giu nguyen cong suat giong noi. Ngoai dai, duong cam thap hon nhieu — nhieu bi suy giam manh.
>
> Vung to do nhat la phan bi bo loc loai bo. Co the thay hieu qua loc rat ro rang tren bieu do nay.

---

## Slide 10: Signal-to-Noise Ratio (SNR)

> Day la truc quan hoa viec phan tach tin hieu va nhieu. Hinh tren la tin hieu tho. Hinh giua la phan duoc giu lai sau loc — chinh la giong noi. Hinh duoi la phan bi loai bo — chinh la nhieu.
>
> SNR duoc tinh bang cong thuc 10 log cua ti so cong suat tin hieu tren cong suat nhieu. SNR cang cao nghia la tin hieu cang sach. Ket qua cho thay bo loc FIR giu lai phan lon nang luong giong noi va loai bo nhieu hieu qua.

---

## Slide 11: Pre-emphasis

> Sau bo loc FIR, nhom ap dung them pre-emphasis. Cong thuc rat don gian: lay mau hien tai tru di 0.97 lan mau truoc do.
>
> Bieu do ben trai la pho sau FIR — tan so cao van yeu hon tan so thap. Bieu do ben phai la sau khi them pre-emphasis — pho da phang hon dang ke, tan so cao duoc tang cuong.
>
> Tai sao can lam viec nay? Vi trong giong noi tu nhien, nang luong giam khoang 6 dB moi octave. Phu am va sibilant bi yeu hon nhieu so voi nguyen am. Pre-emphasis can bang lai, giup buoc trich xuat MFCC tiep theo hieu qua hon.

---

## Slide 12: Spectrum Comparison Across Speakers

> Day la pho tan so cua 4 speakers sau khi loc FIR. Cuong, Quang, Anne va Khoa — moi nguoi co mot pho tan so KHAC NHAU.
>
> Su khac biet nay den tu cau truc thanh quan — moi nguoi co vi tri formant F1, F2, F3 khac nhau. Day chinh la "van tay giong noi" — dau van tay am thanh cua moi nguoi.
>
> MFCC se nam bat su khac biet nay va chuyen thanh vector so de SVM phan lop. Neu khong co bo loc FIR, nhieu se lam mo su khac biet giua cac speakers.

---

## Slide 13: Feature Engineering

> Day la so sanh 2 phuong phap trich xuat dac trung.
>
> Ben trai la Pipeline A — baseline. Chi trich 6 dac trung thoi gian don gian: RMS Energy trung binh va do lech chuan — do do lon tieng noi. ZCR trung binh va do lech chuan — do so lan tin hieu doi dau. Va bien do trung binh va do lech chuan. Van de la 6 dac trung nay chi mo ta mien thoi gian, hoan toan khong co thong tin tan so.
>
> Ben phai la Pipeline B — MFCC. Quy trinh 6 buoc: chia tin hieu thanh frame, ap dung cua so Hamming, tinh FFT, dua qua Mel filterbank mo phong cach tai nguoi nghe, lay log, roi ap dung DCT de co 13 he so MFCC moi frame. Cuoi cung tinh mean va std cua 13 MFCC — duoc vector 26 chieu. Vector nay giau thong tin tan so hon nhieu so voi 6 dac trung cua Pipeline A.

---

## Slide 14: Feature Visualization

> Day la truc quan hoa dac trung tu notebook.
>
> Ben trai la MFCC heatmap — 13 he so MFCC thay doi theo thoi gian cho 1 file audio. Mau sac the hien gia tri cua tung he so. Cac ban co the thay MFCC nam bat duoc su bien thien cua giong noi theo thoi gian.
>
> Ben phai la feature scatter plot — chieu cac dac trung xuong 2 chieu. Cac speakers tao thanh cac cum rieng biet — dac biet voi MFCC features cua Pipeline B, cac cum tach biet ro hon. Day la ly do SVM co the phan lop tot voi Pipeline B.

---

## Slide 15: SVM with RBF Kernel

> Ca hai pipeline deu dung cung mot classifier: SVM voi kernel RBF.
>
> Tai sao chon SVM? Vi no hoat dong tot voi dataset nho — chi 125 mau. No hieu qua trong khong gian nhieu chieu — 26 dim cua MFCC. Va no tao ra decision boundary ro rang.
>
> Kernel RBF do tuong dong giua 2 diem dua tren khoang cach Euclidean, voi tham so gamma kiem soat tam anh huong. C la tham so dieu chinh giua viec phan lop dung va tong quat hoa.
>
> Training pipeline gom StandardScaler normalize features trong moi fold de tranh data leakage, roi SVM phan lop. Nhom dung GridSearchCV voi 3-fold ben trong de tim C va gamma tot nhat, va 5-fold Stratified CV ben ngoai de danh gia. Stratified dam bao ti le moi speaker dong deu trong moi fold.

---

## Slide 16: Cross-Validation Results

> Day la bieu do ket qua cross-validation tu notebook 03_train.
>
> Pipeline A — duong xanh — dao dong quanh muc 56%, rat thap va khong on dinh. Pipeline B — duong cam — dat 97% voi do lech chuan chi 3.6%. Su khac biet giua hai pipeline la rat lon va nhat quan qua tat ca 5 fold.
>
> Bar chart ben phai cho thay ro: Pipeline A dat 56.3% con Pipeline B dat 97.0% — chenh lech 40.7 diem phan tram.

---

## Slide 17: Experimental Results (Table)

> Day la bang tong hop ket qua. Ca hai pipeline deu chon duoc cung best C = 1 va best gamma = scale.
>
> Pipeline A dat accuracy 56.3% voi khoang tin cay 52 den 60%. F1-score la 0.69.
> Pipeline B dat 97.0% voi khoang tin cay 92 den 100%. F1-score la 1.0 — tuc la perfect precision va recall.
>
> Quan trong nhat la ket qua paired t-test: t = -34.79, p = 0.000004. Gia tri p nho hon 0.05 rat nhieu, xac nhan su khac biet la co y nghia thong ke cao — khong phai do ngau nhien.

---

## Slide 18: Confusion Matrices

> Day la confusion matrix tu notebook 04_evaluation.
>
> Pipeline A ben trai — nhieu o ngoai duong cheo co gia tri lon, nghia la model nham lan nhieu giua cac speakers. Vi du Cuong bi nham thanh Quang, Anne bi nham thanh Khoa.
>
> Pipeline B ben phai — gan nhu toan bo gia tri nam tren duong cheo chinh, nghia la model nhan dien dung gan nhu tat ca. Day la bang chung truc quan cho thay Pipeline B vuot troi.

---

## Slide 19: ROC Curves (One-vs-Rest)

> Day la duong cong ROC cho tung speaker theo phuong phap one-vs-rest.
>
> Pipeline A ben trai — cac duong cong nam xa duong cheo, AUC khong cao. Pipeline B ben phai — tat ca cac duong cong sat goc tren ben trai, AUC gan bang 1.0 cho moi speaker.
>
> AUC = 1.0 nghia la model co kha nang phan biet hoan hao giua speaker do va cac speaker con lai. Pipeline B dat duoc dieu nay cho hau het cac speakers.

---

## Slide 20: Performance Comparison

> Day la bar chart so sanh 4 metric chinh giua 2 pipeline: Accuracy, Precision, Recall va F1-score.
>
> Co the thay Pipeline B vuot troi tren tat ca cac metric. Dac biet la Accuracy chenh lech tu 56% len 97%, va F1-score tu 0.69 len 1.0. Ket qua nay chung to rang DSP preprocessing khong chi cai thien accuracy ma con cai thien toan dien chat luong phan lop.

---

## Slide 21: Why Pipeline B Wins

> Tong ket 3 ly do chinh tai sao Pipeline B thang:
>
> Thu nhat — do giau dac trung. 26 chieu MFCC nam bat duoc hinh dang thanh quan, bao gom vi tri cac formant va cach chung thay doi theo thoi gian. Con 6 chieu cua Pipeline A chi co thong tin bien do — khong du de phan biet nguoi noi.
>
> Thu hai — khu nhieu. Bo loc FIR loai bo nhieu ngoai dai tan giong noi. Pre-emphasis can bang nang luong giua tan so thap va cao. Ket qua la SNR tang dang ke truoc khi trich xuat features, giup MFCC sach hon.
>
> Thu ba — hieu suat. Toan bo cac buoc DSP them vao chi mat duoi 1 mili giay cho moi clip 3 giay tren phan cung hien dai. Nhung mang lai 40.7 diem phan tram do chinh xac — mot su danh doi cuc ky hieu qua.

---

## Slide 22: Limitations

> Nhom cung nhan ra mot so han che cua du an:
>
> Dataset con nho — chi 5 nguoi, 125 mau. Ket qua co the khong tong quat duoc cho so luong nguoi noi lon hon hoac da dang hon.
>
> Du lieu thu trong moi truong yen tinh. Khi test bang mic trong lop on thi do chinh xac giam dang ke. Ly do la nhieu nen trong lop — tieng nguoi noi chuyen — nam trong cung dai tan 300 den 3400 Hz voi giong noi, nen bo loc FIR khong loc duoc. Nhom da them thu vien noisereduce de cai thien nhung van chua toi uu.
>
> He thong la closed-set — chi nhan dien 5 nguoi da biet, khong the tu choi nguoi la. Neu mot nguoi moi noi vao, model van se gan cho 1 trong 5 speakers.
>
> Nhom chi thu nghiem SVM — cac mo hinh khac nhu Random Forest hay neural network co the cho ket qua khac.
>
> Va dai tan bo loc 300 den 3400 Hz la co dinh theo tieu chuan, chua duoc toi uu hoa rieng cho dataset nay.

---

## Slide 23: Conclusion

> Tom lai, nhom rut ra 3 ket luan chinh:
>
> Mot — DSP preprocessing la thiet yeu cho bai toan nhan dien nguoi noi. Tang 40.7 diem phan tram do chinh xac, duoc xac nhan boi paired t-test voi p = 0.000004 — co y nghia thong ke rat cao.
>
> Hai — dac trung MFCC 26 chieu nam bat dac trung thanh quan tot hon nhieu so voi 6 dac trung thoi gian co ban. MFCC chua thong tin tan so — chinh la yeu to then chot de phan biet giong noi.
>
> Ba — DSP thu cong ket hop SVM dat 97% tren dataset nho ma khong can deep learning. Dieu nay cho thay trong cac ung dung thuc te voi du lieu han che va yeu cau real-time, handcrafted DSP van rat gia tri.
>
> Huong phat trien tuong lai: mo rong so nguoi noi, them noise reduction cho mic thuc te trong moi truong on, va tich hop DSP frontend voi cac kien truc mang neural.

---

## Slide 24: Live Demo

> Bay gio em se demo truc tiep ung dung. Ung dung viet bang Streamlit, cho phep 2 cach test:
>
> Cach 1 — upload file WAV da thu san. Cach nay cho ket qua chinh xac nhat vi file co cung dac tinh voi training data.
>
> Cach 2 — thu am truc tiep tu mic. Nhom da them noise reduction de xu ly nhieu moi truong.
>
> [Mo app tai localhost:8502]
> [Upload 1 file WAV cua Cuong — cho thay prediction dung voi confidence cao]
> [Upload 1 file cua Anne — cho thay model nhan dien dung nguoi khac]
> [Mo phan "Phan tich tin hieu" — giai thich waveform raw vs filtered, MFCC plot]
> [Neu con thoi gian: thu mic de demo noise reduction]

---

## Slide 25: Thank You

> Cam on thay va cac ban da lang nghe. Nhom san sang tra loi cau hoi.

---

# Cau hoi thuong gap (FAQ) — chuan bi san

**Q: Tai sao chon SVM ma khong dung deep learning?**
> Vi dataset chi co 125 mau — qua nho cho deep learning. Deep learning can hang ngan den hang trieu mau de train. SVM hoat dong tot voi du lieu nho va khong gian nhieu chieu. Voi DSP giup trich xuat feature tot, SVM da du manh de dat 97%.

**Q: Tai sao Pipeline A dat 56% chu khong phai random (20%)?**
> 56% van tot hon random 20% vi RMS va ZCR van chua mot it thong tin phan biet — vi du giong nam va nu co RMS khac nhau. Nhung 6 dac trung nay khong du de phan biet chi tiet giua 5 nguoi.

**Q: Co the dung model nay cho tieng Viet va tieng Anh khong?**
> Co. MFCC trich xuat dac trung thanh quan — khong phu thuoc ngon ngu. Thanh quan cua moi nguoi giong nhau bat ke noi tieng gi. Tuy nhien, training data nen da dang ve noi dung de model khong overfit vao cau noi cu the.

**Q: Mic thu am trong lop khong chinh xac?**
> Vi model train tren audio sach, nhieu lop hoc nam trong dai tan 300-3400 Hz nen FIR khong loc duoc. Nhom da them noise reduction bang thu vien noisereduce — dung spectral gating de uoc luong va tru noise. Tuy nhien de chinh xac nhat, nen upload file wav.

**Q: C va gamma trong SVM la gi?**
> C la tham so regularization — C lon thi model co gang phan lop dung tat ca diem nhung de overfit, C nho thi cho phep sai so nhung tong quat hon. Gamma kiem soat tam anh huong cua moi diem — gamma lon thi chi quan tam diem rat gan (de overfit), gamma nho thi xet pham vi rong hon. Nhom dung GridSearchCV de tu dong tim bo C, gamma tot nhat.

**Q: 97% co phai overfit khong?**
> Khong hoan toan. 97% la ket qua cua 5-fold cross-validation — nghia la moi fold test tren du lieu chua thay. Tuy nhien, F1 tren training set dat 1.0 cho thay co risk overfit. Voi dataset lon hon va da dang hon, con so nay co the giam. Day la 1 limitation nhom da ghi nhan.
