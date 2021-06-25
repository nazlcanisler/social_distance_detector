# USAGE
	# cmd open 
	# python main.py

# gerekli paketleri içe aktar
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2

#non-maxima uygulanırken eşik değer ile birlikte zayıf algılananları
#filtrelemek için min olasılık üzerinden başlatıyorum
MIN_CONF = 0.3
NMS_THRESH = 0.3

# NVIDIA CUDA GPU'nun kullanılması gerekip gerekmediğini gösteren boole değeri
USE_GPU = True

# iki kişinin birbirinden uzak olabileceği min güvenli mesafe tanımı
MIN_DISTANCE = 50

def detect_people(frame, net, ln, personIdx=0):
	# çerçenin boyutlarını alıyoruz ve sonuç değişkenine boş liste atanır
	(H, W) = frame.shape[:2]
	results = []

	# giriş çerçevesinden bir blob oluşturulur ve ardından sınırlayıcı kutuları ve 
	# ilişkinlendiriğimiz YOLO paketlerinin özellikleri girilir.
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# sırasıyla tespit edilen sınırlayıcı kutular
	# centroidler ve güven listelerimizi başlatın
	boxes = []
	centroids = []
	confidences = []

	# katman çıktılarının her biri üzerinde döngü
	for output in layerOutputs:
		# algılamaların her biri üzerinde döngü yapın
		for detection in output:
			# sınıf kimliğini ve güvenini (yani olasılık) ayıklayın)
			#geçerli nesne algılama
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# algılamaları (1) tespit edilen nesnenin bir kişi olduğundan emin olarak filtreleyin
			# ve min güvene göre ayarlanır
			if classID == personIdx and confidence > 0.3:
				# sınırlayıcı kutunun koordinatlarını görüntünün boyutuna göre ölçeklendirin, 
				#YOLO NUN aslında sınırlayıcı kutunun merkezini (x, y) 
				#ve ardından kutuların genişliğini ve yüksekliğini döndürdüğünü unutmayın
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# sınırlayıcı kutunun üst ve sol köşesini türetmek için
				# merkez (x, y) koordinatlarını kullanın
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# sınırlayıcı kutu koordinatları, centroidler ve güvenler listemizi güncelleyin
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# zayıf, üst üste binen sınırlayıcı kutuları bastırmak için 
	#maksimum olmayan bastırma uygulayın
	indices = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# en az bir algılama olduğundan emin olun
	if len(indices) > 0:
		# ltuttuğumuz dizinler üzerinde döngü
		for i in indices.flatten():
			#sınırlayıcı kutu koordinatlarını ayıkla 
			box = boxes[i]
			x = box[0]
			y = box[1]
			w = box[2]
			h = box[3] 

			# sonuç listemizi, kişi tahmin olasılığı,
			# sınırlayıcı kutu koordinatları ve centroid'den oluşacak şekilde güncelleyin
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of results
	return results

# argümanı inşa et ve argümanları ayrıştır
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="./test.mp4",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# YOLO modelimizin eğitim aldığı COCO sınıfı etiketlerini yükleyin
labelsPath ='coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# YOLO ağırlıklarına ve model yapılandırmasına giden yolları türetme
weightsPath = "yolov3-tiny.weights"
configPath = "yolov3-tiny.cfg"

# COCO veri setinde eğitilmiş YOLO nesne dedektörümüzü yükleyin(80 sınıf)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# GPU kullanıp kullanmayacağımızı kontrol edin
if USE_GPU:
	# sCUDA'yı tercih edilen arka uç ve hedef olarak ayarlayın
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# YOLO'dan yalnızca ihtiyacımız olan *çıktı* katman adlarını belirleyin
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# video dosyasının çıktısını almak için video akışını ve işaretçiyi başlat
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# video akışından kareler üzerinde döngü
while True:
	# dosyadan sonraki kareyi oku
	(grabbed, frame) = vs.read()

	# çerçeve yakalanmadıysa, akışın sonuna ulaştık
	if not grabbed:
		break

	# çerçeveyi yeniden boyutlandırın ve ardından içindeki insanları 
	# ve yalnızca insanları) tespit edin
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# minimum sosyal mesafeyi ihlal eden dizinler kümesini başlat
	violate = set()

	# *en az* iki kişi tespiti olduğundan emin olun 
	#(ikili mesafe haritalarımızı hesaplamak için gereklidir)
	if len(results) >= 2:
		# sonuçlardan tüm centroidleri çıkarın ve 
		#tüm centroid çiftleri arasındaki Öklid mesafelerini hesaplayın
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# mesafe matrisinin üst üçgeni üzerinde döngü
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# herhangi iki merkez çifti arasındaki 
				# mesafenin yapılandırılmış piksel 
				# sayısından az olup olmadığını kontrol edin
				if D[i, j] < 50:
					# ihlal setimizi centroid çiftlerinin indeksleri ile güncelleyin
					violate.add(i)
					violate.add(j)

	# sonuçlar üzerinde döngü
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# sınırlayıcı kutuyu ve merkez koordinatlarını çıkarın, 
		# ardından açıklamanın rengini başlatın
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# dizin çifti ihlal kümesi içinde mevcutsa, rengi güncelleyin
		if i in violate:
			color = (0, 0, 255)

		# (1) kişinin çevresine bir sınırlayıcı kutu ve 
		# (2) kişinin merkez koordinatlarını çizin,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# çıktı çerçevesine toplam sosyal mesafe ihlali sayısını çizin
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# çıktı çerçevesinin ekranınızda görüntülenip görüntülenmeyeceğini kontrol edin
	if args["display"] > 0:
		#çıktı çerçevesini göster
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# 'e' ("e"xit) tuşuna basılmışsa, döngüden çık
		if key == ord("e"): 
			break

	# bir çıkış video dosyası yolu sağlanmışsa ve video yazıcı başlatılmamışsa, bunu şimdi yapın
	if args["output"] != "" and writer is None:
		# video yazıcımızı başlat
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# video yazıcısı Yok değilse, çerçeveyi çıkış video dosyasına yazın
	if writer is not None:
		writer.write(frame)

	

