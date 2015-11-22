(ns ai-project.core
  (:gen-class))
(use 'clojure.repl) ; for doc
(use 'criterium.core) ; benchmarking
(use 'hnetwork.core) ; neural network

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

(def msong1 
  "reordered dataset" 
  (i/$ [ :artist_familiarity
				 :duration
				 :end_of_fade_in
				 :key
				 :loudness
				 :mode
				 :start_of_fade_out
				 :tempo
				 :time_signature
				 :year
				 :artist_hotttnesss
				 :song_hotttnesss] msong))

(def msong2 
  "vector version, reordered dataset" 
  (i/to-vect msong1))

(def msong3
  "filtered vector dataset"
  (let [f1 (filterv (fn [row] 
                      (every? #(not= "nan" %) row)) msong2)
        f2 (filterv #(not= 0.0 (peek %)) f1)] 
    f2))

(def msongv 
  "scaled vector dataset" 
 (bias (norm-scale (bias msong3))))

(defn hn [input output samples alpha]
  (/ samples (* alpha (+ input output))))

(defn cnt [] 
  (concat (list (- (count (first msongv)) 1)) (list 16 1)))

(let [verbose false
      tsize 50
      shuffled-set (shuffle (shuffle msongv))
      test-set (into [] (take tsize shuffled-set))
      training-set (into [] (take 500 (drop tsize shuffled-set)))]
        (def w
          "adjusted weights for msongv with nifty-feeder"
          (nifty-feeder training-set 5 [0.5 0.1] (cnt) :verbose-flag verbose))

        (when verbose
          (println "Final Weights")
          (println "w1") (pmm (first w))
          (println "w2") (pmm (last w)))
            
        (def ec 
          "count errors"
          (feed test-set w 0.1 :training false))
        
        (let [ymean (second ec)
              yV (last ec)]
          
        (def sse
          "Error Sum of Squares,
            Measure of the variation of the observed values around the regression line."
          (reduce + (mapv #(let [[y yhat ycost] %1]
                               (Math/pow ycost 2)) yV)))
        
        (def tss
          "SST Total Sum of Squares,
          Measure of the variation of the observed values around the mean."
          (reduce + (mapv #(let [[y yhat ycost] %1]
                               (Math/pow (- ycost ymean) 2)) yV)))        
        (def variance (/ tss tsize))
        
        (def standard-error (Math/sqrt variance))
      
        (println "\nTotal Error -" (first ec))
        (println "\nAverage Error -" (* 100.0 ymean))
        (println "\nSSE -" sse) 
        (println "\nTSS -" tss)
        (println "\nVariance -"variance)
        (println "\nStandard-Error -"standard-error)
        
        (when verbose
          (println " [y        yhat    |y-yhat| ]")
          (pmm (last ec)))))

(defn -main
  "ANN to predict hotness of a song, sgd optimization"
  [& args])
