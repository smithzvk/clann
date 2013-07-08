
(in-package :clann)

(defun sigmoid (x) (/ (+ 1 (exp (- x)))))

(defun forward-propagation (inputs network)
  (let ((nl (network-list network))
        (z* nil)
        (a* (list inputs)))
    (cons
     (first a*)
     (iter (for (th units) :in nl)
       (setf z* (bm:* (unmap-into
                       'array
                       (append-imas
                        (list (make-list (ima-dimension (first a*) 0)
                                         :initial-element '(1))
                              (first a*))
                        1))
                      (ima:transpose th))
             a* (collecting (ima:map-ima 'sigmoid (unmap-into 'array z*))))))))
