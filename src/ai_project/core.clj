(ns ai-project.core
  (:gen-class))
(use 'clojure.repl) ; for doc
(use 'criterium.core) ; benchmarking
(use 'clojure.core.matrix) ; math
(use 'clojure.math.combinatorics) ; math

(set-current-implementation :vectorz); matrix computations
(require '[clojure.java.io :as io]); io resources
(require '[incanter.core :as i]); statistics library
(require '[incanter.datasets :as ds]); datasets, get-dataset
(require '[incanter.excel :as xls]); excel
(require '[incanter.stats :as stat]); stats
(require '[incanter.charts :as chart]); charts
(require '[incanter.io :as iio]); csv

(def msong 
  "unchanged dataset" 
  (iio/read-dataset (str (io/resource "msong.csv")) :header true))

(defn l2v
  "convert list to vector matrix"
  [matrix]
  (mapv #(into [] %) matrix))

(defn gen-matrix
  "generates a `r` by `c` matrix with random weights between -1 and 1."
	[r c & m]
	(for [_ (take r (range))] 
   (for [_ (take c (range))] 
     (* (if (< 0.5 (rand)) -1 1) (rand)))))

(defn weight-gen 
  "generates a multitiered matrix"
  [lst]
  (loop [acc (transient []) t lst]
    (if (= 1 (count t))
           (persistent! acc)
           (recur (conj! acc (l2v (apply gen-matrix t))) (drop 1 t)))))

(defn max-fold
  "max values for a matrix"
  [fn lst]
  (loop [acc (transient []) t lst]
    (if (every? empty? t)
      (rseq (persistent! acc))
      (recur (conj! acc (apply fn (map peek t))) (map pop t))))); handle scaling for negative attributes

(defn bias [lst]
  (mapv (fn [x] (mapv #(+ % 0.001) x)) lst))

(defn scaled 
  [x mn mx]
  (/ (- x mn) (- mx mn)))

(defn norm-scale
  "scales values in matrix to a range between -1 and 1, utilizing max-fld"
  [lst]
  (let [blst (bias lst)
        mx (max-fold max blst)
        mn (max-fold min blst)]
        (println "asdf" mx "jkl;" mn )
        (mapv (fn [x] (mapv scaled x mn mx)) blst)))


(defn sigmoid
  "takes in `z` and throws it in the sigmoid function\n"
  [z]
    (/ 1 (+ 1 (Math/exp (* -1 z)))))

(defn mmap
  "maps a function on a weight vector matrix"
  [function matrix]
  (mapv #(mapv function %) matrix))

(defn multitiered-forward
  "takes in weights `w and inputs `x and propagates the inputs though the network"
  [input w]
  (loop [x input weights w]
    (if (empty? weights)
      x
      (recur 
         (mmap sigmoid (dot x (first weights))) ; first weights -> weights in this later so
         (rest weights)))))

(defn pluck
  "extract a value from nexted matrix"
  [fn matrix]
  (fn (first matrix)))

(defn adjust-weights
  "feeds data into nn and returns adjusted weights"
  [x w y lr]
  (let [z (pluck first (dot x w))
        yhat (sigmoid z)
        xt (transpose x); [[x1 x2 x3   to [[x1] [x2] [x3]]
        ycost (* -1 (- y yhat)); -(y-yhat)
        enz (Math/exp (* -1 z)); e^(-z)
        sigmoid-prime (/ enz (Math/pow (+ 1 enz) 2)); enz/(1+enz)^2
        delta-w (mmap #(* (* ycost sigmoid-prime) %) xt)
        lrdw (mmap #(* % lr) delta-w)
        wkp1 (i/minus w lrdw)]
    wkp1))

(defn feed
  "loops across input and adjustes the weights for all of it. 
  `input` assumes y values are at the end of the vectors"
  [input weight learnrate]
  (loop [x input w weight]
    (if (every? empty? x)
      w
      (recur (pop x) (let [thisx (peek x)
                           in [(pop thisx)]
                           out (peek thisx) ] 
                       (adjust-weights in w out learnrate))))))

(defn feed-one 
  "feeds a single `x` into the ANN"
  [x w]
  (let [z (pluck first (dot x w))
        yhat (sigmoid z)]
    yhat))

(defn find-error
  "find the error given a `value` and `theta` and if is what we expect to `match`"
  [yhat y]
    (abs (- yhat y)))

(defn error-check
  "checks error given `inputs` `weights` `threshold`"
  [input weight]
  (loop [in input er 0.0 acc []]
    (if (every? empty? in)
      [er (/ er (count input)) acc]
      (let [row (peek in)
            x [(pop row)]
            y (peek row)
            yhat (feed-one x weight)
            fr (find-error yhat y)]
        (recur (pop in) (+ er fr)
          (conj acc [yhat y (- y yhat)]))))))

(defn error-loop
  "step between min and max and error-check to examine outliers."
  [mn mx step data weight]
  (loop [m mn acc []]
    (if (> (- m step) mx)
      acc
      (recur (+ step m) 
             (conj acc [m (first (error-check data weight m))])))))

(defn refeed 
  "refeeds results from training at different learning rates `lrs`"
  [data weight lrs]
  (loop [w weight lr lrs]
    (if (empty? lr)
      w
      (recur (feed data w (first lr)) (rest lr)))))

(defn expand
  "expands the dataset for testing"
  [dataset magnitude]
  (loop [ds dataset m (- magnitude 1)]
    (if (<= m 0)
      (shuffle (shuffle ds))
      (recur (into [] (concat ds dataset)) (- m 1) ))))

(defn nifty-feeder
  "expands and feeds a dataset, useful for finding that special rate"
  [data magnitude lrs size  & {:keys [verbose-flag] :or {verbose-flag false}}]
  (let [dt (expand data magnitude)
        w (first (weight-gen size))
        w2 (refeed dt w lrs)
        ]
    (when verbose-flag (println "Initial Weights") (pm w))
     w2))

(defn bestset 
  [sets]
  (loop [acc (transient []) bs sets]
    (println "sets left -" (count bs))
    (if (empty? bs)
      (persistent! acc)
      (let [attributes (conj (first bs) :sex)
            ds1 (i/$ attributes msong)
            dsv (i/to-vect ds1)
            dss (norm-scale dsv)
            lst (concat (list (- (count (first dss)) 1)) (list 1))
            wi (nifty-feeder dss 180 [0.12 0.06 0.01] lst)
            e1 (error-check dss wi 0.5)
            e2 (error-check dss wi 0.41)
            ]
        (recur (conj! acc [(first e1) (first e2) attributes]) (rest bs))))))

(def msong1 
  "reordered dataset" 
  (i/$ [ :artist_familiarity
				 :artist_playmeid
				 :duration
				 :end_of_fade_in
				 :key
				 :key_confidence
				 :loudness
				 :mode
				 :mode_confidence
				 :start_of_fade_out
				 :tempo
				 :time_signature
				 :time_signature_confidence
				 :year
				 :artist_hotttnesss
				 :song_hotttnesss] msong))

(def msong2 
  "vector version, reordered dataset" 
  (i/to-vect msong1))

(def msong3
  "filtered vector dataset"
  (filterv (fn [row] 
             (every? #(not= "nan" %) row)) msong2))
 
(def msongv 
  "scaled vector dataset" 
  (norm-scale msong3))

(defn cnt [] 
  (concat (list (- (count (first msongv)) 1)) (list 1)))

; (def ssets (into [] (rest (mapv #(into [] %) (subsets [:sp :FL :RW :CL :CW :BD])))))

(def w
  "adjusted weights for msongv with nifty-feeder"
   (nifty-feeder msongv 30 [0.1] (cnt) :verbose-flag [true]))


(let [ec (error-check msongv w)
      er (first ec)
      ac (last ec)
      ep (second ec)]
  (println "\nFinal Weights")
  (pm w)
  (println "\nError -" er)
  (println "Err % -" ep))
  ; (println "\nResults with errors:\n[Result][Off By]")
  ; (pm ac)

(defn -main
  "ANN to predict hotness of a song, sgd optimization"
  [& args])
