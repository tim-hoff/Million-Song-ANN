(defproject ai-project "0.1.0"
  :description "Hotness prediction with sgd"
  :url "http://github.com/tim-hoff/ai-project"
  :license {:name "None"}
  :dependencies [[org.clojure/clojure "1.8.0-beta2"]
              [org.clojure/core.async "0.1.346.0-17112a-alpha"]
							[criterium "0.4.3"]
              [net.mikera/vectorz-clj "0.36.0"]
							[net.mikera/core.matrix "0.42.1"]
      				[incanter/incanter "1.9.0"]]
  :plugins [[lein-codox "0.9.0"]]
  :main ^:skip-aot ai-project.core
  :target-path "target/%s"
  :jvm-opts ["-Xmx2g" "-server"]
  :profiles {:uberjar {:aot :all}})
