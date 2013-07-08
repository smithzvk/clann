
(in-package :clann)

(defun logistic (x) (/ (+ 1 (exp (- x)))))

(defun add-biases (arr)
  (unmap-into
   'array
   (append-imas
    (list (make-list (ima-dimension arr 0)
                     :initial-element '(1))
          arr)
    1)))

(defun forward-propagation (inputs network)
  (let ((nl (network-list network))
        (z nil)
        (a (list (add-biases inputs))))
    (cons
     (first a)
     (iter (for (th units) :in nl)
       (setf z (bm:* (first a) (ima:transpose th))
             a (collecting
                (add-biases (ima:map-ima 'logistic (unmap-into 'array z)))))))))
