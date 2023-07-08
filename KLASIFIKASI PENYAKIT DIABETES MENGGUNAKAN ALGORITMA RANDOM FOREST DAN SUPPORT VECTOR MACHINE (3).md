# KLASIFIKASI PENYAKIT DIABETES MENGGUNAKAN ALGORITMA RANDOM FOREST DAN SUPPORT VECTOR MACHINE


### DOMAIN PROJEK
Penyakit diabetes merupakan salah satu penyakit yang berbahaya di dunia. Menurut World Health Organization pada tahun 2020, penyakit diabetes masuk dalam sepuluh besar penyebab utama kematian secara global.Menurut _International Diabetes Federation_ (IDF), prevalensi penyakit diabetes di dunia pada tahun 2021 sebanyak 536,6 juta, dan akan meningkat sebanyak 11,3% menjadi 642,7 juta pada tahun 2030, dan 12,2% menjadi 783,2 juta pada tahun 2045. Selain itu, Indonesia diprediksi akan menempati ranking kelima dimana sebanyak 28,6 juta penduduk Indonesia akan terkena penyakit ini pada tahun 2045.
Walaupun tidak menular, penyakit diabetes dapat menyerang siapa saja. Oleh karena itu, masyarakat perlu untuk mewaspadai adanya kemungkinan terkena penyakit ini. Oleh karena itu, dengan data yang ada, kemungkinan terjadinya penyakit diabetes dapat diketahui dengan cepat. Data yang ada dapat dimanfaatkan secara maksimal, salah satunya yaitu klasifikasi dalam machine learning. Klasifikasi dapat didefinisikan sebagai teknik yang mempelajari tentang sekumpulan data sehingga mendapatkan aturan yang dapat mengenali data baru yang belum dipelajari sebelumnya. Penelitian dengan data penyakit diabetes juga telah dilakukan oleh Nugraha & Sabaruddin (2021). Dataset yang digunakan yaitu _Pima Indians Diabetes_, dengan total variabel sebanyak 9, dimana 1 variabelnya merupakan output/label. Dari penelitian tersebut didapatkan hasil bahwa dengan metode Random Forest akurasinya sebesar 75,82% (_oversampling_), 71,24% (_undersampling_), dan 73,86 (original). Penelitian lainnya juga telah dilakukan oleh Lyngdoh et al., (2020) menggunakan data diabetes dengan tujuh variabel dan beberapa algoritma seperti _K-Nearest Neigbour, Naïve Bayes_, dan _Decision Tree_. Dari penelitian itu didapatkan kesimpulan bahwa _K-Nearest Neighbor_ menghasilkan akurasi yang paling baik diantara algoritma yang lainnya, yaitu sebesar 76%. Akan tetapi, keakuratan tersebut belum terlalu baik.

## _BUSINESS UNDERSTANDING_
Dalam prediksi diagnosa, machine learning merupakan metode yang menjanjikan untuk pengembangan deteksi penyakit. Metode tersebut disusun sehingga mampu mengeksplorasi sebuah data, menemukan sebuah pola, dan membantu menemukan sebuah pengetahuan yang baru. Data yang ada dapat dimanfaatkan secara maksimal, salah satunya yaitu klasifikasi. Klasifikasi merupakan metode pengelompokan data yang memiliki kelas atau target. Menurut Sutoyo dan Fadlurrahman (2020), klasifikasi merupakan salah satu salah satu fungsi dari data mining untuk mengelompokkan suatu item data ke dalam kategori atau kelas yang telah didefinisikan. Klasifikasi ini dilakukan dengan tujuan untuk memperkirakan kelas dari suatu objek yang labelnya belum diketahui sebelumnya.Cara kerjanya yaitu algoritma klasifikasi mempelajari data training yang berisikan data dan label yang telah diklasifikasikan terlebih dahulu. Algoritma berisi pembelajaran akan belajar dengan data yang ada dan menemukan sebuah hubungan antara data input dan output/label. Selanjutnya algoritma melakukan klasifikasi pada data testing yang hanya berisi data input tanpa label berdasarkan pengalaman mempelajari data training. Algoritma yang populer dalam klasifikasi adalah algoritma _Random Forest. Random Forest_ merupakan salah satu metode klasifikasi yang terdiri dari kumpulan pohon keputusan. Random Forest memiliki keunggulan  dapat memberikan akurasi yang tinggi. Selain _Random Forest_, terdapat algoritma _Support Vector Machine_ yang juga bisa digunakan dalam klasifikasi. _Support Vector Machine_ memiliki kelebihan dimana algoritma ini mempunyai kemampuan generalisasi yang tinggi dan dapat menghasilkan model klasifikasi yang baik. SVM memiliki kelebihan seperti dapat bekerja mengklasifikasikan data yang linier ataupun nonlinier. Algoritma SVM dapat melakukan generalisasi dengan klasifikasi data lain yang tidak termasuk ke dalam data yang digunakan.

### _Problem Statements_
1.	Bagaimana perbandingan klasifikasi penyakit diabetes menggunakan algoritma Random Forest dan Support Vector Machine (SVM)?
2.	Bagaimana performa Random Forest dan Support Vector Machine (SVM) dalam klasifikasi penyakit diabetes, serta manakah yang lebih baik?
3. Variabel apa yang paling berpengaruh dalam klasifikasi penyakit diabetes ini?

### _Goals_
1.	Membandingkan klasifikasi penyakit diabetes menggunakan algoritma _Random Forest_ dan _Support Vector Machine_ (SVM).
2.	Menentukan performansi hasil klasifikasi penyakit diabetes menggunakan algoritma _Random Forest_ dan _Support Vector Machine_ (SVM).
3.	Mengetahui  variabel yang paling berpengaruh dalam klasifikasi penyakit diabetes ini.

## _Data Understanding_
Data yang digunakan dalam projek ini merupakan data sekunder dari Rumah Sakit Shyllet, Bangladesh pada tahun 2020 yang diperoleh melalui situs Kaggle. Data ini berjumlah sebanyak 520, dan  memiliki 17 variabel, di mana 1 variabelnya merupakan label. Dari data ini terdapat 200 data tidak menderita penyakit diabetes dan 320 data menderita penyakit diabetes. Data ini diambil melalui situs Kaggle (_https://www.kaggle.com/datasets/alakaaay/diabetes-uci-dataset?resource=download_)

Variabel yang ada pada data yaitu:
1) _Age_: 20-65
2) _Sex: Male/Female_
3) _Polyuria: Yes/No_
4) _Polydipsia: Yes/No_
5) _sudden weight loss: Yes/No_
6) _weakness: Yes/No_
7) _Polyphagia: Yes/No_
8) _Genital thrush: Yes/No_
9) _visual blurring: Yes/No_
10) _Itching: Yes/No_
11) _Irritability: Yes/No_
12) _delayed healing: Yes/No_
13) _partial paresis: Yes/No_
14)_muscle stiffness: Yes/No_
15) _Alopecia: Yes/No_
16) _Obesity: Yes/No_
17) _Class: Positive/Negative_


## _Data Preparation_
_CEK MISSING VALUE._
```sh
data.isna().sum()
```
Dari pengecekan tersebut tidak ada _missing value_.

Selanjutnya yaitu melakukan _label encoding_. Data yang bernilai “tidak”, “Laki-Laki”, dan “Negatif”  akan diubah dengan label “-1”, dan data yang bernilai “Ya”, “Perempuan”, dan “Positif” akan diubah menjadi “1”. Label encoding dilakukan karena algoritma support vector machine merupakan algoritmma yang basisnya menggunakan jarak, sehingga data kategorik harus diubah menjadi numerik, selain itu pada python dalam klasifikasi tidak dapat mengolah data kategorik.

Langkah selanjutnya yaitu melihat apakah terdapat outliers pada data. Dari semua variabel yang ada, terdapat outliers pada variabel usia (karena data lainnya hanya berisi 2 tipe, ya/tidak, atau laki laki/perempuan). Outliers dapat dilihat melalui Gambar dibawah ini. 

## CEK _OUTLIERS_

![image](https://github.com/powtoonang/coba1/blob/main/outliers.png)



Gambar 1. _Outliers_

Outliers pada variabel Age ini tidak dibuang. Hal itu dikarenakan outliers tersebut merupakan fenomena dari subjek penelitian, sehingga tidak dilakukan pembuangan data.

Langkah selanjutnya yaitu dilakukan pengecekan keseimbangan data.

![image](https://github.com/powtoonang/coba1/blob/main/2.%20keseimbangan%20data.png)


Gambar 2. Keseimbangan Data

Gambar tersebut memperlihatkan bahwa data tidak seimbang. Oleh karena itu, perlu dilakukan penanganan keseimbangan data. Data dapat diseimbangkan menggunakan _Synthetic Minority Oversampling Technique_ (SMOTE). Data tersebut diseimbangkan setelah data di split menjadi data training dan data testing. Rasio pembagian data training dan data testing sebesar 80% dan 20%.

## MODELLING
Karena data _imbalance_, maka dilakukan penyeimbangan data. Metode yang digunakan yaitu Synthetic Minority Oversampling Technique (SMOTE). Hasil data yang telah diseimbangkan dapat dilihat dibawah ini.

![image](https://github.com/powtoonang/coba1/blob/main/3.%20setelah%20smote.png)


Gambar 3. Keseimbangn Data Setelah SMOTE

terlihat bahwa data sudah seimbang. Data dengan label -1 setelah diseimbangkan menjadi sebanyak 230 data

### Klasifikasi Menggunakan Algoritma _Random Forest_
Algoritma _Random Forest_ merupakan salah satu _ensemble learning_. _Ensemble learning_ adalah metode dimana model akan dilatih untuk memecahkan masalah yang sama dan digabungkan untuk mendapatkan suatu hasil yang lebih baik. Algoritma _Random Forest_ ini ialah algoritma yang dikembangkan dari algoritma _Decision Tree_. _Decision tree_ adalah algoritma yang berbentuk sebuah pohon untuk mengambil kesimpulan. _Decision Tree_ ini dapat digunakan untuk mengklasifikasikan sebuah data dengan variabel input dan output dalam bentuk pohon. Terdapat beberapa istilah yang digunakan dalam _Decision Tree_, yaitu _root node_, _internal node_, dan _leaf_. Root merupakan _node_ yang terletak pada bagian paling atas di pohon. Internal node merupakan node percabangan yang masih memiliki cabang di bawahnya, sedangkan leaf merupakan node akhir yang tidak memiliki percabangan lagi. _Decision Tree_ akan memasukkan sebuah input melalui _root_, dan memiliki kesimpulan melalui leaf node untuk menentukan data input termasuk dalam kelas yang mana. Algoritma ini dikembangkan menjadi sebuah algoritma baru yang dinamakan sebagai Random Forest. Sesuai dengan namanya, algoritma ini akan menciptakan sebuah hutan dengan sejumlah pohon. Cara kerja klasifikasi menggunakan algoritma ini yaitu Random Forest akan melakukan bootstrap pada data training untuk membentuk setiap pohon. Selanjutnya, pohon tersebut akan digabungkan dengan pohon yang lain, dimana satu pohon akan menghasilkan satu keputusan. Oleh sebab itu, algorima _Random Forest_ ini dapat dikatakan sebagai kumpulan _Decision Tree_. Untuk mendapatkan hasil akhir, maka dilakukan _majority voting_ dimana _vote_ terbanyak akan menjadi pemenangnya. Menurut Pamuji dan Ramadhan (2021), algoritma _Random Forest_ ini memiliki kelebihan yaitu dapat menghasilkan eror yang relatif rendah, performa yang baik dalam klasifikasi, dan cocok untuk data yang berjumlah besar. Parameter yang dapat digunakan dalam algoritma ini yaitu _n estimator_ (jumlah pohon), _max feature_ (jumlah variabel yang perlu dipertimbangkan saat mencari pemisah terbaik), _max depth_ (kedalaman pohon), dan lain lain.

Hasil kinerja dari Random Forest dapat dilihat melalui gambar ini.
![image](https://github.com/powtoonang/coba1/blob/main/4.%20cf%20rf.png)


Gambar 4. _Confusion Matrix Random Forest_


_Feature Importance Random Forest_
_feature importance Random Forest_ dapat dilihat melalui gambar dibawah ini

![image](https://github.com/powtoonang/coba1/blob/main/5.%20f%20importance.png)


Gambar 5. _Feature Importance Random Forest_



### KLASIFIKASI MENGGUNAKAN ALGORITMA _SUPPORT VECTOR MACHINE_
Support vector machine (SVM) merupakan salah satu algoritma yang dapat digunakan dalam klasifikasi. SVM menggunakan _hyperplane_ dengan pemisah dari pengelompokan kelas yang dibentuk dari suatu dimensi vektor berukuran n (Iman dan Wijayanto, 2021). Hyperplane terbaik didapatkan dengan mengukur margin _hyperplane_ atau jarak antara vektor yang paling dekat dengan _hyperplane_. 

Pada SVM dilakukan normalisasi data. Hal ini dilakukan karena svm bekerja menggunakan jarak, dimana data akan dipisahkan menggunakan _hyperplane_. Terdapat variabel usia yang memiliki nilai besar dibandingkan dengan variabel yang lainnya. Oleh karena itu dilakukan normalisasi data. 
Hasil klasifikasi menggunakan SVM dapat dilihat melalui confusion matrix dibawah ini.
![image](https://github.com/powtoonang/coba1/blob/main/6.%20cf%20svm.png)

Gambar 6. _Confusion Matrix SVM_


# EVALUASI
Berdasarkan hasil klasifikasi, akan dilakukan perbandingan dari kedua model tersebut. Melalui _confusion matrix_, dapat dihitung _evaluation matrix_ untuk menilai kinerja dari sebuah algoritma klasifikasi, yaitu akurasi, presisi, _recall, dan f1 score_.
_Evaluation matrix_ dari _Random forest_ dapat dilihat melalui gambar dibawah ini. 

![cfrf](https://github.com/powtoonang/coba1/blob/main/4.%20cf%20rf.png)

Gambar 7. _Evaluation Matrix Random Forest_

Dari gambar tersebut dapat diketahui bahwa akurasi, presisi, _recall_, dan _f1 score_ dari _random forest_ sebesar 0,99. Sedangkan untuk SVM _evaluation matrix_ dapat dilihat melalui gambar dibawah ini.

![cfsvm](https://github.com/powtoonang/coba1/blob/main/6.%20cf%20svm.png)

Gambar 8. Evaluasi Matrix SVM

Dari gambar tersebut dapat diketahui bahwa akurasi, presisi, recall, dan f1 score dari svm sebesar 0,98.

Berdasarkan hasil analisis yang telah dilakukan, didapatkan kesimpulan bahwa:
1. Algoritma _Random Forest_ dan _Support Vector Machine_ mampu mengklasifikasikan data penyakit diabetes dengan baik.
2. Algoritma _Random Forest_ terbaik memiliki akurasi yang didapatkan sebesar 0,98, presisi 0,96, _recall_ 1, _specificity_ 0,95, dan _F1-score_ sebesar 0,98. Pada algoritma _Support Vector Machine_, model terbaiknya yaitu berada pada split data 80%:20% dengan akurasi 0,92, presisi 0,89, _recall_ 0,98, _specificity_ 0,84, dan _F1-score_ 0,93. Dari kedua algoritma didapatkan bahwa _Random Forest_ dengan split data 80%:20% merupakan algoritma terbaik dalam klasifikasi penyakit diabetes ini.
3. Tiga variabel yang paling berpengaruh dalam klasifikasi penyakit diabetes ini secara berturut turut yaitu _polyuria, polydipsia_, dan jenis kelamin.

Dari pernyataan diatas diketahui bahwa kesimpulan tersebut sudah menjawab _problem statement_ yang telah didefinisikan sebelumnya. Untuk mengembangkan penelitian ini, dapat dicoba menggunakan model boosting apakah model tersebut lebih baik atau tidak, sehingga dapat ditemukan sebuah model klasifikasi yang dapat dikembangkan lebih jauh lagi dalam prediksi penyakkit diabetes ini.

# REFERENSI
[1] IDF. (2021). IDF Diabetes Atlas 10th Edition. www.diabetesatlas.org diakses pada tanggal 8 Agustus 2022.

[2] Iman, Q. & Wijayanto, A. W. (2021). Klasifikasi Rumah Tangga Penerima Beras Miskin (Raskin)/Beras Sejahtera (Rastra) di Provinsi Jawa Barat Tahun 2017 dengan Metode Random Forest dan Support Vector Machine. JUSTIN (Jurnal Sistem dan Teknologi Informasi), 9(2), 178-184.

[3] Lyngdoh, A. C., Choudhury, N. A., & Moulik, S. (2021, March). Diabetes Disease Prediction Using Machine Learning Algorithms. In 2020 IEEE-EMBS Conference on Biomedical Engineering and Sciences (IECBES) (pp. 517-521). IEEE.

[4] Nugraha, W., & Sabaruddin, R. (2021). Teknik Resampling untuk Mengatasi Ketidakseimbangan Kelas pada Klasifikasi Penyakit Diabetes Menggunakan C4. 5, Random Forest, dan SVM. Techno. Com, 20(3), 352-361.

[5] Pamuji, F. Y. & Ramadhan, V. P. (2021). Komparasi Algoritma Random Forest dan Decision Tree untuk Memprediksi Keberhasilan Immunotheraphy. Jurnal Teknologi dan Manajemen Informatika, 7(1), 46-50.

[6] Sutoyo, E., & Fadlurrahman, M. A. (2020). Penerapan SMOTE untuk Mengatasi Imbalance Class dalam Klasifikasi Television Advertisement Performance Rating Menggunakan Artificial Neural Network. JEPIN (Jurnal Edukasi dan Penelitian Informatika), 6(3), 379-385.

[7] WHO. (2020). The Top 10 Causes of Death. Diakses dari https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death 

[8] WHO. (2021). Diabetes. Diakses dari  https://www.who.int/health-topics/diabetes#tab=tab_1 

[9] WHO. (2022). Diabetes. Diakses dari https://www.who.int/news-room/fact-sheets/detail/diabetes
