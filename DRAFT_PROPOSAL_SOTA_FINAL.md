

<!-- Start of picture text -->
iANUDD,<br>% \y<br><!-- End of picture text -->

## BAB I **PENDAHULUAN** 

### **1.1 Latar Belakang** 

Rumput laut jenis Kappaphycus dan Eucheuma merupakan komoditas budidaya laut tropis yang bernilai ekonomi tinggi, terutama sebagai bahan baku karagenan yang banyak dipakai industri pangan, farmasi, dan kosmetik. Budidaya jenis rumput laut ini banyak dilakukan di kawasan Asia Tenggara termasuk Indonesia, dan menjadi sumber penghidupan penting bagi masyarakat pesisir. Namun produktivitas budidaya ini kerap terganggu oleh penyakit ice-ice, yaitu kondisi memutihnya thallus yang diikuti kerusakan jaringan akibat interaksi antara perubahan kondisi lingkungan seperti suhu dan salinitas dengan aktivitas bakteri oportunistik, yang pada akhirnya menyebabkan hilangnya biomassa dan penurunan hasil panen (Ward et al., 2022). 

Upaya mendeteksi ice-ice secara otomatis melalui citra sudah mulai dikembangkan. Salah satu penelitian terbaru menerapkan algoritma K-Nearest Neighbor pada citra rumput laut dengan memanfaatkan fitur warna dan tekstur Gray Level Cooccurrence Matrix, dan berhasil mencapai akurasi keseluruhan 86,67 persen dari 44 pengujian pada dataset sebanyak 400 hingga 600 citra (Saputro et al., 2024). Pendekatan ini menunjukkan bahwa klasifikasi berbasis citra memang dapat membantu petani rumput laut mengenali penyakit lebih cepat, tetapi masih bergantung pada fitur tangan yang kurang mampu menggeneralisasi variasi kondisi lapangan, dan tetap memerlukan ratusan citra per kelas agar performanya layak. 

Kebutuhan akan ratusan citra ini menjadi masalah tersendiri di lapangan, karena penyebaran ice-ice pada satu petak budidaya tidak selalu merata dan pengambilan gambar bawah air sering terkendala visibilitas serta cuaca. Few-shot learning muncul sebagai pendekatan yang relevan untuk kondisi seperti ini, karena dirancang agar model dapat mengenali kelas baru hanya dari beberapa contoh saja. Salah satu metode few-shot learning berbasis metrik yang paling banyak dipakai adalah Prototypical Network, yang bekerja dengan membentuk satu titik representasi atau prototipe untuk tiap kelas dari rata-rata fitur beberapa contoh yang tersedia, kemudian mengklasifikasikan data baru berdasarkan kedekatan jaraknya 

ke tiap prototipe tersebut (Snell et al., 2017). Dalam beberapa tahun terakhir, pendekatan berbasis prototipe seperti ini juga sudah mulai dijajaki pada domain alga, misalnya untuk klasifikasi spesies mikroalga air tawar di Macau (Wang et al., 2023) dan klasifikasi mikroalga laut menggunakan penguatan fitur multi skala (Liu et al., 2025), maupun pada domain pertanian untuk klasifikasi penyakit tanaman dengan data minim (Rezaei et al., 2024). 

Meskipun demikian, ProtoNet standar memiliki kelemahan mendasar yaitu prototipe dihitung dengan cara merata-ratakan semua citra support secara setara. Ketika sebagian citra yang tersedia justru berkualitas buruk akibat pencahayaan bawah air yang tidak merata, gerakan kamera, atau kekeruhan air, prototipe yang terbentuk ikut bergeser dari representasi kelas yang sebenarnya. Isu ketahanan prototipe pada skenario data terbatas ini memang sudah menjadi perhatian dalam literatur few-shot learning beberapa tahun terakhir, misalnya melalui agregasi fitur yang lebih tahan terhadap sampel menyesatkan sebagai pengganti rata-rata sederhana (Liang et al., 2022) dan mekanisme rektifikasi prototipe berbasis selfattention yang menyesuaikan kontribusi tiap sampel secara adaptif terhadap kelasnya (Zhao et al., 2024). Namun kedua pendekatan ini menurunkan bobot sampel berdasarkan sinyal dari dalam ruang fitur atau label saja, tanpa secara eksplisit mempertimbangkan bahwa sebagian sampel yang dipakai untuk membentuk prototipe memang secara kualitas citra kurang layak dijadikan acuan. 

Di sisi lain, bidang penilaian kualitas citra tanpa referensi atau No-Reference Image Quality Assessment sudah berkembang cukup matang untuk mengukur tingkat degradasi sebuah citra tanpa memerlukan citra pembanding yang bersih, misalnya melalui pendekatan self-supervised yang mempelajari manifold distorsi citra (Agnolucci et al., 2024) maupun pendekatan berbasis transfer learning yang memadukan fitur semantik global dan lokal untuk mengatasi keterbatasan data pelatihan (Yang et al., 2025). Khusus untuk citra bawah air seperti pada studi kasus penelitian ini, degradasi yang muncul memiliki karakteristik khas berupa hamburan cahaya dan distorsi warna akibat media air, sehingga memerlukan metrik penilaian kualitas yang disesuaikan, sebagaimana diusulkan melalui metrik Underwater Image Fidelity (Zheng et al., 2022). 

Bertolak dari kondisi ini, penelitian ini mengusulkan Quality-Aware Prototypical Network atau QA-ProtoNet, yaitu modifikasi terhadap ProtoNet standar yang mengganti mekanisme rata-rata sederhana dengan rata-rata berbobot, di mana bobot tiap citra ditentukan dari perpaduan skor kualitas citra tanpa referensi dan tingkat tipikalitasnya dalam ruang fitur. Studi kasus yang dipakai adalah klasifikasi status sehat dan terjangkit ice-ice pada citra lapangan rumput laut, dengan harapan hasil penelitian ini dapat menjadi alat bantu deteksi dini yang tetap andal meski data pelatihan dan kualitas citra yang tersedia terbatas. 

### **1.2 Rumusan Masalah** 

Bagaimana rancangan mekanisme pembobotan prototipe berbasis keandalan citra, yang memadukan penilaian kualitas citra tanpa referensi dengan tipikalitas ruang fitur, dapat meningkatkan ketahanan dan kemampuan generalisasi few-shot learning terhadap degradasi citra lapangan dibandingkan dengan Prototypical Network standar dan metode prototipe robust berbasis ruang fitur lainnya? 

## **1.3 Hipotesis** 

QA-ProtoNet, yang membentuk prototipe kelas melalui rata-rata berbobot berdasarkan fusi skor kualitas citra tanpa referensi dan skor tipikalitas ruang fitur, diduga menghasilkan akurasi klasifikasi yang lebih tinggi dan lebih stabil dibandingkan Prototypical Network standar maupun metode prototipe robust berbasis ruang fitur lainnya, khususnya pada subset citra rumput laut yang mengalami degradasi akibat kondisi pengambilan di lapangan. Semakin besar proporsi citra terdegradasi dalam support set, semakin besar pula selisih performa yang diharapkan muncul antara QA-ProtoNet dan ProtoNet standar. 

### **1.4 Manfaat Penelitian** 

Dari aspek akademik, penelitian ini memberikan kontribusi berupa framework baru yang mengisi celah riset pada pertemuan tiga bidang aktif yaitu surrogate-assisted optimization, particle swarm intelligence, dan multi-fidelity 

HPO untuk neural network. Kontribusi ini berpotensi menghasilkan publikasi di jurnal Scopus Q1 atau Q2 dalam bidang Swarm and Evolutionary Computation, Applied Soft Computing, atau IEEE Transactions on Evolutionary Computation. Dari aspek praktis, framework yang dihasilkan berpotensi mengurangi waktu dan biaya komputasi pencarian hyperparameter secara signifikan, yang pada gilirannya mendukung demokratisasi deep learning bagi peneliti dan praktisi dengan sumber daya komputasi terbatas. 

### **1.5 Tujuan Penelitian** 

1. Merancang mekanisme pembobotan prototipe yang memadukan penilaian kualitas citra tanpa referensi dengan tipikalitas ruang fitur pada arsitektur Prototypical Network. 

2. Mengevaluasi performa QA-ProtoNet dibandingkan dengan Prototypical Network standar dan metode prototipe robust berbasis ruang fitur lainnya, khususnya pada subset citra yang terdegradasi. 

3. Menguji kemampuan generalisasi mekanisme pembobotan yang diusulkan pada domain citra lain di luar studi kasus rumput laut. 

### **1.6 Manfaat Penelitian** 

Secara teoritis, penelitian ini diharapkan memberi kontribusi metodologis pada bidang few-shot learning, khususnya dalam bentuk mekanisme reliability-weighted prototyping yang dapat diadaptasi ke domain citra lapangan lain yang memiliki masalah serupa, yaitu data terbatas sekaligus kualitas citra yang bervariasi. 

Secara praktis, penelitian ini diharapkan dapat menjadi dasar pengembangan alat bantu deteksi dini penyakit ice-ice bagi petani rumput laut, khususnya di wilayah yang pengumpulan data citranya sulit dilakukan dalam jumlah besar. 

### **1.7 Batasan Penelitian** 

1. Klasifikasi dibatasi pada dua kelas, yaitu rumput laut sehat dan rumput laut yang terjangkit ice-ice.Citra yang digunakan adalah citra lapangan 

2. yang diambil langsung pada kondisi budidaya nyata, bukan citra hasil akuisisi laboratorium dengan kondisi pencahayaan terkontrol. 

3. Penelitian ini berfokus pada perancangan mekanisme pembobotan prototipe, bukan pada tahap deteksi maupun segmentasi objek rumput laut dalam citra. 

4. Evaluasi performa dilakukan pada skema few-shot dengan jumlah shot terbatas, mengikuti protokol episodic training yang umum dipakai pada literatur few-shot learning. 

### **1.6 State-of-the-Art Penelitian Terdahulu** 

Tabel berikut merangkum sepuluh penelitian terkait yang diterbitkan dalam lima tahun terakhir (2022–2025) dan menjadi dasar penentuan posisi penelitian ini. 

|**Peneliti**<br>**(Tahun)**|**Metode**|**Objek/Domain**|**Kelebihan**|**Keterbatasan /**<br>**Gap**|
|---|---|---|---|---|
|Ward dkk.<br>(2022)|Tinjauan<br>sistematik faktor<br>lingkungan dan<br>mikrobiologis|Epidemiologi<br>ice-ice pada<br>eucheumatoid|Memetakan<br>penyebab dan<br>dampak ekonomi<br>ice-ice secara<br>komprehensif|Tidak membahas<br>pendekatan citra<br>atau komputasi|
|Saputro dkk.<br>(2024)|KNN dengan<br>fitur warna dan<br>GLCM|Klasifikasi ice-<br>ice rumput laut|Akurasi 86,67%<br>pada citra<br>lapangan|Fitur tangan, butuh<br>ratusan citra per<br>kelas, sensitif<br>pencahayaan|
|sWang dkk.<br>(2023)|Few-shot learning<br>dan transfer<br>learning|Klasifikasi<br>spesies<br>mikroalga|Efektif pada data<br>mikroskopis<br>terbatas|Fokus spesies,<br>bukan status<br>penyakit, tidak<br>mempertimbangkan<br>kualitas citra|
|Liu dkk.<br>(2025)|AlgaeClass_Net,<br>few-shot dengan<br>penguatan fitur<br>multi skala|Klasifikasi<br>mikroalga laut|Akurasi 91,20%<br>pada skema 5-way<br>5-shot|Fokus fitur<br>diskriminatif,<br>belum<br>mempertimbangkan<br>keandalan sampel|
|Rezaei dkk.<br>(2024)|ResNet18<br>dipadukan<br>Prototypical<br>Network|Klasifikasi<br>penyakit<br>tanaman data<br>minim|Akurasi 93% pada<br>dataset<br>PlantVillage|Rata-rata prototipe<br>standar, tanpa<br>pembobotan<br>kualitas|
|Liang dkk.<br>(2022)|Agregasi fitur<br>robust (median,<br>similarity<br>weighting) dan<br>TraNFS|Few-shot<br>learning<br>dengan label<br>noise|Prototipe lebih<br>tahan terhadap<br>sampel<br>menyesatkan|Pembobotan<br>berdasar label/fitur,<br>bukan skor kualitas<br>citra eksplisit|
|Zhao dkk.<br>(2024)|Rektifikasi<br>prototipe dengan<br>self-attention|Klasifikasi<br>few-shot umum|Kontribusi tiap<br>sampel<br>disesuaikan adaptif<br>per relevansi kelas|Belum melibatkan<br>penilaian kualitas<br>citra sebagai sinyal<br>pembobotan|
|Agnolucci<br>dkk. (2024)|ARNIQA, NR-<br>IQA self-<br>supervised|Penilaian<br>kualitas citra<br>umum|Skor kualitas<br>selaras persepsi<br>manusia tanpa<br>label besar|Belum diterapkan<br>pada few-shot<br>learning atau<br>akuakultur|
|Yang dkk.<br>(2025)|Transfer learning<br>dan fusi fitur<br>adaptif (IQA-<br>NRTL)|Penilaian<br>kualitas citra<br>umum|Mengatasi<br>keterbatasan<br>ukuran dataset<br>IQA publik|Belum terhubung<br>ke pembentukan<br>prototipe few-shot|



|**Peneliti**<br>**(Tahun)**|**Metode**|**Objek/Domain**|**Kelebihan**|**Keterbatasan /**<br>**Gap**|
|---|---|---|---|---|
|Zheng dkk.<br>(2022)|UIF, metrik<br>fidelitas citra<br>bawah air|Penilaian<br>kualitas citra<br>bawah air|Menilai keaslian<br>warna, ketajaman,<br>dan struktur citra<br>bawah air|Belum<br>diintegrasikan ke<br>pipeline few-shot<br>learning|



# **DAFTAR PUSTAKA** 

<mark>Agnolucci, L., Galteri, L., Bertini, M., & Del Bimbo, A. (2024). ARNIQA: Learning distortion manifold for image quality assessment. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 189–198. https://doi.org/10.1109/WACV57701.2024.00026</mark> 

<mark>Liang, K. J., Rangrej, S. B., Petrovic, V., & Hassner, T. (2022). Few-shot learning with noisy labels. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9089–9098. https://doi.org/10.1109/CVPR52688.2022.00888</mark> 

<mark>Liu, D., Yuan, G., Tan, H., Jiang, Y., Bi, H., & Cheng, Y. (2025). AlgaeClass_Net: Optimizing few-shot marine microalgae classification with multi-scale feature enhancement network. IEEE Access, 13, 16223–16237. https://doi.org/10.1109/ACCESS.2024.3436838</mark> 

<mark>Rezaei, M., Diepeveen, D., Laga, H., Jones, M. G. K., & Sohel, F. (2024). Plant disease recognition in a low data scenario using few-shot learning. Computers and Electronics in Agriculture, 219, 108812. https://doi.org/10.1016/j.compag.2024.108812</mark> 

<mark>Saputro, A. K., Ibadillah, A. F., Rahman, Alfita, R., Purnamasari, D. N., & Laksono, D. T. (2024). Analysis of ice-ice disease on seaweed using the K- Nearest Neighbor algorithm in vision robot technology. Proceedings of the 2024 IEEE 10th Information Technology International Seminar (ITIS), 108 –114. https://doi.org/10.1109/ITIS64716.2024.10845286</mark> 

<mark>Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. Advances in Neural Information Processing Systems, 30, 4077– 4087.</mark> 

<mark>Wang, B., Yuan, A., Zou, H., Chen, Z., & Li, J. (2023). Algae species classification based on few-shot learning: A case study of Macau freshwater. Proceedings of the 2023 9th International Conference on Communication and Information Processing (ICCIP), 133–137. https://doi.org/10.1145/3638884.3638904</mark> 

<mark>Ward, G. M., Kambey, C. S. B., Faisan, J. P., Tan, P. L., Daumich, C. C., Matoju, I., Stentiford, G. D., Bass, D., Lim, P. E., Brodie, J., & Poong, S. W. (2022).</mark> 

<mark>Ice-ice disease: An environmentally and microbiologically driven syndrome in tropical seaweed aquaculture. Reviews in Aquaculture, 14(1), 414–439. https://doi.org/10.1111/raq.12606</mark> 

<mark>Yang, Y., Liu, C., Wu, H., & Yu, D. (2025). A quality assessment algorithm for noreference images based on transfer learning. PeerJ Computer Science, 11, e2654. https://doi.org/10.7717/peerj-cs.2654</mark> 

<mark>Zhao, P., Wang, L., Zhao, X., Liu, H., & Ji, X. (2024). Few-shot learning based on prototype rectification with a self-attention mechanism. Expert Systems with Applications, 249, 123586. https://doi.org/10.1016/j.eswa.2024.123586 Zheng, Y., Chen, W., Lin, R., Zhao, T., & Le Callet, P. (2022). UIF: An objective quality assessment for underwater image enhancement. IEEE Transactions on Image Processing, 31, 5456–5468. https://doi.org/10.1109/TIP.2022.3196815</mark> 

