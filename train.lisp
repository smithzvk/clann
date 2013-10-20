
(in-package :clann)

(defun gradient-descent (inputs network outputs
                         &key
                         (step-size 0.001)
                         (max-steps 1000))
  (iter (for i :below max-steps)
    (bind:bind (((:values cost gradient) (cost inputs network outputs)))
      (collect cost)
      (let ((nl (network-list network)))
        (iter (for (th unit) :in nl)
          (for grad :in gradient)
          (setf (self-map th)
                (map-ima (lambda (x y)
                           (- x (* step-size y)))
                         th grad)))))))
