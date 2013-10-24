
(in-package :clann)

;;; These are some examples for testing purposes

;; Simple NOT network

(defparameter *not* (make-network '(1 1)))

(predict '((1) (0)) *not*)
;; #2A((0.5022051428419341d0) (0.5006472352011774d0))

(gradient-descent '((1) (0)) *not* '((0) (1)) :step-size .1)

(predict '((1) (0)) *not*)
;; => #2A((0.04178159530190728d0) (0.9370432509543996d0))

;; One of many NOT network

(defparameter *not-oom* (make-network '(2 2)))

(predict '((1 0) (0 1)) *not-oom*)
;; => #2A((0.999283159678386d0 1.0036827267310713d0)
;;        (1.0114637840828975d0 1.0062172800889833d0))

(gradient-descent '((1 0) (0 1)) *not-oom* '((0 1) (1 0)) :step-size 0.1)

(predict '((1 0) (0 1)) *not-oom*)
;; => #2A((0.013276724079558641d0 0.9932924985137532d0)
;;        (0.9932939825950216d0 0.01327963225376837d0))

;; Superfluous layer

(defparameter *not-extra* (make-network '(1 1 1)))

(predict '((1) (0)) *not-extra*)
;; => #2A((0.4993879561712247d0) (0.4993877344429867d0))

;; More steps for more layers
(gradient-descent '((1) (0)) *not-extra* '((0) (1)) :step-size .1 :max-steps 2000)

(predict '((1) (0)) *not-extra*)
;; => #2A((0.004455518674449654d0) (0.9934938513089426d0))

;; OOM extra layer

(defparameter *not-oom-extra* (make-network '(2 2 2)))

(predict '((1 0) (0 1)) *not-oom-extra*)
;; => #2A((0.999283159678386d0 1.0036827267310713d0)
;;        (1.0114637840828975d0 1.0062172800889833d0))

(gradient-descent '((1 0) (0 1)) *not-oom-extra* '((0 1) (1 0)) :step-size .1)

(predict '((1 0) (0 1)) *not-oom-extra*)
;; => #2A((0.013276724079558641d0 0.9932924985137532d0)
;;        (0.9932939825950216d0 0.01327963225376837d0))




;; OR network

(defparameter *or* (make-network '(2 1)))

(predict '((1 1) (0 0) (1 0) (0 1)) *or*)
;; #2A((0.49809242463615805d0)
;;     (0.49938955247632183d0)
;;     (0.49875730829712883d0)
;;     (0.49872466458132425d0))

(gradient-descent '((1 1) (0 0) (1 0) (0 1)) *or* '((1) (0) (1) (1))
                  :step-size .1 :max-steps 10000)

(predict '((1 1) (0 0) (1 0) (0 1)) *or*)
;; #2A((0.9999985665966035d0)
;;     (0.02053382116812035d0)
;;     (0.9917989732272424d0)
;;     (0.9917989812801093d0))




