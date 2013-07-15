
(in-package :clann)

(defun gradient-descent (inputs network outputs
                         &key
                         (batch-size (ima-dimension inputs 0))
                         (step-size 0.001)
                         (max-steps 1000))
  (iter (for i :below max-steps)
    (let ((gradient (back-propagation inputs network outputs batch-size))
          (nl (network-list network)))
      (iter (for (th unit) :in nl)
        (for grad :in gradient)
        (setf (self-map th)
              (map-ima (lambda (x y)
                         (- x (* step-size y)))
                       th grad))))))
