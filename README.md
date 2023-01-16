# Laporan Proyek Machine Learning - Erin Nur Fatimah

## Project Overview

Perpustakaan merupakan elemen yang sangat penting dalam sebuah Perguruan Tinggi/Universitas. Dengan adanya perpustakaan, proses pembelajaran mahasiswa tidak terpaku hanya pada materi yang diajarkan pendidik di kelas. Mahasiswa dapat lebih mendalami materi pelajaran dengan mencari referensi  berupa buku, laporan kerja praktek dan tugas akhir di perpustakaan. Sebuah perpustakaan yang bertaraf internasional sudah pasti memiliki koleksi buku tersebut, terkadang mahasiswa mengalami kesusahan untuk mencari buku yang tepat, yang sesuai dengan yang mereka butuhkan [[1](http://repository.uin-suska.ac.id/1094/)].
Dengan permasalahan di atas tentunya mahasiswa akan terbantu jika ada dosen atau orang lain merekomendasikan sebuah buku yang sesuai dengan topik yang mereka inginkan. Tapi tidak mungkin setiap saat ada orang yang dapat merekomendasikan buku-buku tersebut kepada mahasiswa. Oleh karena itu dalam proyek ini saya membuat sistem rekomendasi (*recommender system*) yang dapat memberikan saran atau rekomendasi buku yang sesuai untuk kebutuhan mereka.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang masalah di atas, terdapat rumusan masalah yang akan diselesaikan: 
- Bagaimana cara merekomendasikan buku yang disukai pengguna lain dapat direkomendasikan kepada pengguna lainnya juga?

### Goals

Berdasarkan rumusan masalah tersebut, tujuan yang akan dicapai yaitu : 
- Membuat sistem rekomendasi yang sesuai dengan preferensi pengguna berdasarkan ratings dan aktifitas pengguna di masa lalu.

### Solution statements
Solusi yang saya buat yaitu dengan menggunakan 2 algoritma Machine Learning sistem rekomendasi, yaitu :
- **Content Based Filtering** adalah algoritma yang merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Content based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.

- **Collaborative Filtering** adalah suatu konsep dimana opini dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai/diminati oleh seorang pengguna. Collaborative filtering memberikan rekomendasi berdasarkan kumpulan dari pendapat, minat dan ketertarikan beberapa user yang biasanya diberikan dalam bentuk rating yang diberikan user kepada suatu item.

## Data Understanding

Data atau dataset yang digunakan pada proyek *machine learning* ini adalah data *Book Recommendation Dataset* yang didapat dari situs kaggle. Link dataset dapat dilihat dari tautan berikut [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Data yang saya gunakan pada *Book Recommendation Dataset* adalah books dan ratings. Masing-masing data tersebut mempunyai jumlah data yang begitu banyak, maka dari itu saya hanya menggunakan beberapa data saja dari total keseluruhan data set yaitu dengan menggunakan data books sebanyak 14000 dan begitu juga ratings sebanyak 14000 data.

- Books : merupakan daftar buku yang tersedia.
- Ratings : merupakan daftar penilaian yang diberikan pengguna terhadap buku. File ini berjumlah 1149780 baris data dan 3 buah kolom yang mencakup User_ID, ISBN, dan Book_Rating.

Lebih lengkapnya, masing-masing dataset di atas berisikan fitur-fitur berikut : 
Books
  * ISBN : merupakan identitas unik suatu buku
  * Book-Title : merupakan judul buku
  * Book-Author : merupakan pengarang buku 
  * Year-Of-Publication : merupakan tahun terbit dipublikasikannya buku
  * Publisher : merupakan penerbit buku
  * Image-URL-S : merupakan gambar dari buku yang berukuran kecil
  * Image-URL-M : merupakan gambar dari buku yang berukuran sedang 
  * Image-URL-L : merupakan gambar dari buku yang berukuran besar 

Rating
  * User-ID : merupakan ID unik pengguna
  * ISBN : merupakan identitas unik suatu buku
  * Book-Rating : merupakan rating dari pengguna yang diberikan untuk penilaian buku

Pada tahapan data ini kita akan mengetahui berbagai macam informasi dataset yang dipakai, diantaranya sebagai berikut :
1.	Langkah pertama yang perlu dilakukan adalah mengimport beberapa library pendukung.
2.	Selanjutnya untuk mengimport dataset agar bisa dilakukan preparation, bisa menggunakan ```pd.read_csv``` dari pandas.
3.  Mengambil data buku dan rating yang digunakan masing-masing sebanyak 14000 dengan fungsi ```iloc()```
4.  Mengubah "-" menjadi "_" agar mempermudah proses pemanggilan
5.	Kita bisa menggunakan ```dt_books.info()``` untuk mengecek tipe kolom pada dataset
6.  Menghapus 3 kolom pada 'data_books' karena tidak akan digunakan dengan memakai fungsi ```drop()``` pada kolom 'Image_URL_S', 'Image_URL_M', 'Image_URL_L'
7.  Mengubah kolom 'Year_Of_Publication' yang semula bertipe object menjadi integer
8.  Melihat data pada variabel rating dengan ```dt_ratings.head()```
9.  ```dt_ratings.describe()``` Untuk melihat distribusi rating.
10. Melihat berapa pengguna yang memberikan rating, jumlah ISBN, dan jumlah rating dengan menggunakan kode seperti berikut :
```
print('Jumlah user_ID: ', len(dt_ratings.User_ID.unique()))
print('Jumlah ISBN: ', len(dt_ratings.ISBN.unique()))
print('Jumlah data rating: ', len(dt_ratings))
```

Selanjutnya kita akan melakukan tahapan Data Preprocesing :
* Menggabungkan dt_books dan dt_ratings
```
dt = dt_ratings.merge(dt_books, left_on = 'ISBN', right_on = 'ISBN')
```
*  ```dt.info()``` untuk mengecek tipe kolom pada data yang baru. Setelah kedua data digabung maka jumlahnya menjadi 7 colom yang terdiri dari 4856 data.
*  Menggunakan barplot untuk menampilkan 10 penulis terpopuler. Untuk lebih detailnya bisa dilihat seperti pada gambar berikut :
![Author](https://i.postimg.cc/7LBYVNrJ/Author.png)
_Gambar 1. Grafik 10 Penulis Terpopuler_
Grafik di atas menampilkan 10 penulis terpopuler yang berasal dari kolom 'Book_Author' dimana peringkat pertama adalah James Patterson.

## Data Preparation
Sebelum menjalankan tahap persiapan data, maka kita perlu melakukan beberapa tahapan data Preparation pada Algoritma **Conten Based Filtering** :
* Menangani Missing Value
Menyeleksi data pada dt_books, dt_rating dan dt (gabungan dari dua data tersebut) apakah data tersebut ada yang kosong atau tidak, jika ada data kosong maka saya akan menghapusnya. Dan dapat kita lihat bahwa tidak terdapat data yang missing value.

* Membuat variabel bernama preparation
Pada tahap ini kita akan melakukan penghapusan data duplikat dengan membuat variable baru bernama ‘preparation’ yang berisi dataframe 'dt’ yang diurutkan berdasarkan ‘ISBN’

* Membuang data duplikat pada variabel preparation dengan fungsi ```drop_duplicates()```
Selanjutnya, kita hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, kita perlu menghapus data yang duplikat dengan fungsi 'drop_duplicates()'. Dalam hal ini, kita membuang data duplikat pada kolom ‘ISBN’. Bisa kita lihat bahwa jumlah rows berkurang ketika dilakukan penghapusan data duplikat. Yang sebelumnya 4856 menjadi 3622 data.

* Melakukan konversi data series menjadi list
Kita perlu melakukan konversi data series ISBN, Title, Author menjadi list. Dalam hal ini, kita menggunakan fungsi 'tolist()' dari library numpy dimana akan menampilkan jumlah dari books_id, books_title dan books_author.

* Membuat dictionary untuk menentukan pasangan key-value pada data books_id, books_title, dan books_author 
Pada tahap berikutnya, kita akan membuat dictionary untuk menentukan pasangan key-value pada data books_id, books_title, dan books_author yang telah kita siapkan sebelumnya.
 
Tahapan Data Preparation pada Algoritma **Collaborative Filtering** :
* Menyandikan (encode) fitur User_ID dan ISBN ke dalam indeks Integer : Digunakan agar model Machine Learning bisa memproses data tersebut.
* Melakukan proses encoding pada ISBN
* Memetakan User_ID dan ISBN ke dataframe yang berkaitan.
Digunakan setelah membuat sebuah data berisi User_ID dan ISBN yang telah menjalani proses encoding lalu kita masukkan kedua fitur tersebut ke dataframe user dan buku.
* Mengecek beberapa hal dalam data seperti jumlah user, jumlah ISBN, dan mengubah nilai ratings menjadi float yang digunakan untuk memudahkan penjumlahan ketika proses pembagian train dan test data.
* Membagi Data untuk Training dan Validasi. 
Digunakan untuk membagi data menjadi data uji dan data latih, dengan proporsi 80:20.

## Modeling
Setelah melakukan tahapan data preparation, langkah selanjutnya adalah memodelkan data. Proses modeling yang saya lakukan pada data ini adalah dengan membuat algoritma machine learning, yaitu content based filtering dan collabrative filtering. 

1. **Content Based Filtering** adalah algoritma yang merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Content based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.

`[+]` Kelebihan dari algoritma ini dapat memberikan rekomendasi item yang belum pernah dirating sekalipun.

`[-]` Kekurangan metode ini adalah tidak dapat merekomendasikan item bagi user baru yang belum pernah melakukan aktivitas apapun.

Berikut ini Proses yang kita lakukan pada **Content Based Filtering**.
1. Menggunakan TF-IDF Vectorizer untuk melakukan pembobotan.
2. Melakukan fit dan transformasi ke dalam bentuk matriks.
3. Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense().
4. Matriks tf-idf untuk beberapa books (title) dan author.
5. Menggunakan fungsi cosine similarity dari library sklearn
6. Membuat dataframe dari variabel cosine_sim_df dengan baris dan kolom berupa author
7. Membuat fungsi bernama ```book_recommendations()```
Sebelum mulai menulis kodenya, kita ingat kembali definisi sistem rekomendasi yang menyatakan bahwa keluaran sistem ini adalah berupa top-N recommendation. Oleh karena itu, kita akan memberikan sejumlah rekomendasi buku pada pengguna yang diatur dalam parameter k.
Dengan menggunakan argpartition, kita mengambil sejumlah nilai k tertinggi dari similarity data (dalam kasus ini: dataframe cosine_sim_df). Kemudian, kita mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Data ini dimasukkan ke dalam variabel closest. Berikutnya, kita perlu menghapus author yang yang dicari agar tidak muncul dalam daftar rekomendasi. Dalam kasus ini, nanti kita akan mencari judul buku kategorinya mirip dengan buku yang ditulis oleh Robert Hendrickson, sehingga kita perlu drop author Robert Hendrickson agar tidak muncul dalam daftar rekomendasi yang diberikan nanti. 

Selanjutnya, mari kita terapkan kode sebelumnya untuk menemukan rekomendasi buku yang mirip dengan kategori buku yang ditulis oleh Robert Hendrickson. Terapkan kode berikut:
```
books_new[books_new.author.eq('Robert Hendrickson')]
```
Output :
|    | id         | title                                             | author             |
|----|------------|---------------------------------------------------|--------------------|
|1465| 1575663937 | More Cunning Than Man: A Social History of Rat... | Robert Hendrickson |

Perhatikanlah, More Cunning Than Man: A Social History adalah buku yang ditulis oleh Robert Hendrickson. Tentu kita berharap rekomendasi yang diberikan adalah buku dengan kategori yang mirip. Nah, sekarang, dapatkan buku recommendation dengan memanggil fungsi yang telah kita definisikan sebelumnya:

```
books_recommendations('Robert Hendrickson')
```
Output :
|    | author                                              | author               |
|----|-----------------------------------------------------|----------------------|
| 1  | Tony Parsons                                        | Man and Boy: A Novel |
| 2  | Tony Parsons                                        | Man and Boy          |
| 3  | Tony Parsons                                        | Man and Boy: A Novel |
| 4  | Tony Parsons                                        | Man and Boy          |
| 5  | Suzanne Simmons                                     | Lady's Man           |

Penjelasan dari hasil rekomendasi di atas yakni kita telah mampu menampilkan 5 buku rekomendasi dengan kategori yang mirip dengan buku yang ditulis oleh Robert Hendrickson.

2.  **Collaborative Filtering** adalah suatu konsep dimana opini dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai/diminati oleh seorang pengguna. Collaborative filtering memberikan rekomendasi berdasarkan kumpulan dari pendapat, minat dan ketertarikan beberapa user yang biasanya diberikan dalam bentuk rating yang diberikan user kepada suatu item.

`[+]` Kelebihan dari pendekantan user based collaborative filtering adalah dapat menghasilkan rekomendasi yang berkualitas baik. 

`[-]` Sedangkan kekurangannya adalah kompleksitas perhitungan akan semakin bertambah seiring dengan bertambahnya pengguna sistem, semakin banyak pengguna (user) yang menggunakan sistem maka proses perekomendasian akan semakin lama.

Berikut ini adalah hasil tampilan output dari 10 rekomendasi buku.
Showing recommendations for users: 277351
|================================================================================|
|--------------------------------------------------------------------------------|
|Top 10 books recommendation                                                     |
|--------------------------------------------------------------------------------|
|The Watsons Go to Birmingham - 1963 (Yearling Newbery) : CHRISTOPHER PAUL CURTIS|
|To Kill a Mockingbird : Harper Lee                                              |
|The Bell Jar : A Novel (Perennial Classics) : Sylvia Plath                      |
|Heat and Dust : Ruth Prawer Jhabvala                                            |
|The Secret Life of Bees : Sue Monk Kidd                                         |
|Rebecca : Daphne Du Maurier                                                     |
|Chicken Soup for the Soul (Chicken Soup for the Soul) : Jack Canfield           |
|Angela's Ashes: A Memoir : Frank McCourt                                        |
|The Demon-Haunted World: Science As a Candle in the Dark : Carl Sagan           |
|Memoirs of a Geisha Uk : Arthur Golden                                          |

Penjelasan dari Hasil rekomendasi di atas yakni kita telah mampu menampilkan 10 buku rekomendasi dengan kategori yang mungkin akan disukai dan belum pernah dipilih/dibaca oleh pengguna sebelumnya menggunakan teknik Collaboratory Filtering.

## Evaluation
1. Evaluasi untuk Model Algoritma **Content Based Filtering**
Disini saya merekomendasikan buku yang ditulis oleh Robert Hendrickson.

Hasil dari Top-N 5 dari film atau movie yang saya rekomendasikan adalah sebagai berikut :

|    | author                                              | title                |
|----|-----------------------------------------------------|----------------------|
| 1  | Tony Parsons                                        | Man and Boy: A Novel |
| 2  | Tony Parsons                                        | Man and Boy          |
| 3  | Tony Parsons                                        | Man and Boy: A Novel |
| 4  | Tony Parsons                                        | Man and Boy          |
| 5  | Suzanne Simmons                                     | Lady's Man           |

Dari hasil rekomendasi di atas, Dari 5 item yang direkomendasikan, 2 item memiliki judul Man and Boy: A Novel, 2 item berujudul Man and Boy, dan satunya lagu berjudul Lady's Man. 

Teknik Evaluasi di atas adalah dengan menggunakan precission. Precision adalah tingkat ketepatan antara informasi yang diminta oleh pengguna dengan jawaban yang diberikan oleh sistem. Adapun rumus dari teknik ini adalah :
```
Precision = ((k of recommendation that are relevant) / (k of item we recommend)) . 100 %
```
Pada contoh rekomendasi buku dapat kita simpulkan bahwa :
k of recommendation that are relevant = 3 buku
k of item we recommend = 5 buku
Precision = ((3)/(5)) . 100 %
Jadi presisinya = 60%

2. Evaluasi untuk Model Algoritma **Collaborative Filtering**
Evaluasi metrik yang digunakan untuk mengukur kinerja model adalah metrik RMSE (Root Mean Squared Error). Root Mean Square Error (RMSE) adalah  metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error [[2](https://www.khoiri.com/2020/12/cara-menghitung-root-mean-square-error-rmse.html)]. 
Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar.
Berikut ini kelebihan dan kekurangan dari RMSE.

`[+]` Kelebihan : menghukum kesalahan besar lebih sehingga bisa lebih tepat dalam beberapa kasus. 

`[-]` Kekurangan : memberikan bobot yang relatif tinggi untuk kesalahan besar. Ini berarti RMSE harus lebih berguna ketika kesalahan besar sangat tidak diinginkan.

Cara Menghitung Root Mean Square Error (RMSE) adalah dengan mengurangi nilai aktual dengan nilai peramalan kemudian dikuadratkan dan dijumlahkan keseluruhan hasilnya kemudian dibagi dengan banyaknya data. Hasil perhitungan tersebut selanjutnya dihitung kembali untuk mencari nilai dari akar kuadrat.

![Metrik RMSE](https://i.postimg.cc/GmMB5sBZ/RMSE.png)
keterangan : 
At = Nilai data Aktual 
Ft = Nilai hasil peramalan
N= banyaknya data
∑ = Summation (Jumlahkan keseluruhan  nilai)

Selanjutnya, cara menerapkan metrik tersebut adalah dengan menambahkan **_'metrics=[tf.keras.metrics.RootMeanSquaredError()]'_** pada model.compile sehingga menjadi seperti berikut :  
```
model = RecommenderNet(num_users, num_book, 50) 
 
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.

Berikut ini adalah plot metrik RMSE setelah proses pelatihan model.  

![Evaluasi Metrik](https://i.postimg.cc/2jdqhtxc/Evaluasi-Metrik.png)

_Gambar 2. Evaluasi Metriks_
Dapat diperhatikan pada hasil proses training model di atas cukup smooth dan model konvergen pada epochs sekitar 25. Dari proses ini, kita memperoleh nilai error akhir sebesar sekitar 0.19 dan error pada data validasi sebesar 0.30. Nilai tersebut cukup bagus untuk sistem rekomendasi.

### Referensi
[1] Handrico. A, “SISTEM REKOMENDASI BUKU PERPUSTAKAAN FAKULTAS SAINS DAN TEKNOLOGI DENGAN METODE COLLABORATIVE FILTERING - Universitas Islam Negeri Sultan Syarif Kasim Riau Repository,” Uin-suska.ac.id, Jan. 2012, doi: http://repository.uin-suska.ac.id/1094/1/2013_201302.pdf.
[2] Khoiri, “Pengertian dan Cara Menghitung Root Mean Square Error (RMSE),” Khoiri.com, Dec. 23, 2020. https://www.khoiri.com/2020/12/cara-menghitung-root-mean-square-error-rmse.html
