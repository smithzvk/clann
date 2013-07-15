
(in-package :clann)

(defun d-logistic (x)
  (let ((val (logistic x)))
    (* val (- 1 val))))
(defun back-propagation (inputs network outputs
                         &optional (batch-size (ima-dimension inputs 0)))
  "Back propagate the derivatives of a batch of inputs through the network."
  (let* ((mb-inputs (submatrix inputs 0 0 batch-size))
         (mb-outputs (submatrix outputs 0 0 batch-size))
         (nl (network-list network))
         (z nil)
         (a mb-inputs))
    (destructuring-bind (as zs)
        ;; forward propagation
        (iter (for (th units) :in nl)
          (setf z (m*m (add-biases a) (ima:transpose th))
                a (ima:map-ima 'logistic (unmap-into 'array z)))
          (collecting z :into zs)
          (collecting a :into as)
          (finally (return (list (reverse (cons mb-inputs as))
                                 (reverse (cons mb-inputs zs))))))
      ;; back propagate
      (reverse
       (iter (for (th units) :in (reverse nl))
         (for (a-prev a) :on as)
         (for z :in (rest zs))
         (for p-delta = (if (first-iteration-p)
                            (map-ima #'- a-prev mb-outputs)
                            delta))
         (for delta = (map-ima #'*
                               (unmap-into 'array (remove-biases (m*m p-delta th)))
                               (map-ima #'d-logistic (unmap-into 'array z))))
         (collecting
          (map-ima (lambda (x) (/ x batch-size))
                   (unmap-into
                    'array
                    (transpose (m:* (transpose (add-biases a)) p-delta))))))))))
