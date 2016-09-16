(def x (range 12))
(def y1 (pow 2 x))
(def y2 (sq x))

(def plot1 (xy-plot x y1 
                    :x-label "Input size (n)" 
                    :y-label "Units of time" 
                    :title "Exponential vs Polynomial"))
(doto plot1 
        (add-pointer 10 150 :text "Efficient" :angle :ne )
        (add-pointer 10 1250 :text "Will take a while"))

(add-lines plot1 x y2)
(view plot1)
