
(in-package :clann)

(defun d-logistic (x)
  (let ((val (logistic x)))
    (* val (- 1 val))))

(defun d-linear (x)
  (declare (ignore x))
  1)

(defun d-rectified-linear (x)
  (if (< x 0)
      0
      1))

;; The <<cost>> function returns the cost of a particular network averaged over
;; a given input set and, as a second value, it's derivative.

;; This setup really only knows how to deal with the "proper cost function".
;; This is the common term for whatever cost function that produces the
;; derivative equal to (activity - output).

;; Square error cost for linear neurons

;; \[ Cost(y) = \Sum_i (t - y)^2 \]

;; Cross entropy for softmax

;; \[ Cost(y,t) = - \Sum_i t \log y \]

;; Whatever that thing that Ng uses that looks like cross entropy but works in
;; cases where the output is a single binary value.

;; \[ Cost(y,t) = - \Sum_i (t \log y - (1 - t) \log (1-y)) \]

;; All of these have derivatives that are simply (a - o).  Any function that you
;; wish to use will need to have this same property.

;; At this time there is not extensible framework to add new cost functions to
;; the system.  This decision was made in order to make the low level code
;; easier to implement.  This isn't very Lispy.

;; Another alternative is to have this calculation happen on the Lisp side of
;; things.  This would allow it to be reasonably flexible at what is probably a
;; significant performance penalty, but not a show stopper.

;; The lowlevel routines forward prop (provided inputs), back prop (provided
;; derivatives).  These are then used with standard Lisp functions to calculate
;; the cost.

;; This way the cost remains on the Lisp side.

;; If the output neuron is linear or rectified linear, that is used.  If the
;; output is logistic, then the Ng cross entropy is used.  If the output is
;; softmax, then the standard cross entropy is used.

;; If no derivative is given, then we assume that the derivative is equal to $(t
;; - y)$.

;; These are only really useful if the entire output is the same.  See if there
;; is something we can do about this.  If costs are always additive (they are 
;; even in the softmax case) then we can do them all separately and combine them
;; at the end.

(defun l2-norm (outputs values)
  "Good for linear outputs."
  (iter
    (for target :in-ima outputs)
    (for y :in-ima (first (last values)))
    (summing (expt (- target y) 2))))

(defun ng-entropy (outputs values)
  "Good for boolean values."
  (iter
    (for target :in-ima outputs)
    (for y :in-ima (first (last values)))
    (summing (- (+ (* target (log y)) (* (- 1 target) (log (- 1 y))))))))

(defun cross-entropy (outputs values)
  "Good for softmax outputs."
  (iter
    (for target :in-ima outputs)
    (for y :in-ima (first (last values)))
    (summing (- (* target (log y))))))

(defun auto-cost (outputs values unit-functions)
  (iter :outer
    (for target-vector :in-column-vectors-of outputs)
    (for y-vector :in-column-vectors-of (first (last values)))
    (iter
      (for fn :in unit-functions)
      (for target :in-ima target-vector)
      (for y :in-ima y-vector)
      (in :outer
          (summing (cond ((member fn (list #'linear 'linear
                                           #'rectified-linear 'rectified-linear))
                          (expt (- target y) 2))
                         ((member fn (list #'logistic
                                           'logistic))
                          (- (+ (* target (log y)) (* (- 1 target) (log (- 1 y))))))
                         ((member fn (list #'softmax
                                           'softmax))
                          (- (* target (log y))))
                         (t (error "Don't know what cost to use for ~A."
                                   fn))))))))

(defun cost (inputs network outputs
             &key (regularization :l2)
                  (regularization-factor 0d0))
  (declare (ignore regularization))
  (bind:bind (((:values a z) (forward-propagation inputs network))
              ((:values cost output-deriv)
               (auto-cost outputs a (second (first (last (network-list network))))))
              (output-deriv (or output-deriv
                                (map-ima #'- (first (last a)) outputs))))
    (let ((gradient (back-propagation inputs network output-deriv a z)))
      (values
       (+ cost
          (* regularization-factor
             (reduce
              #'+
              (mapcar (lambda (theta)
                        (reduce-ima #'+ (map-ima (lambda (x) (expt x 2)) theta)))
                      (mapcar #'alexandria:ensure-car (network-list network))))))
       gradient))))

;; The cost is the summed over the batch (just like the derivative)

;; You have to run forward-prop, then back-prop.  You have to do this by hand as
;; while there are pretty good defaults, there are not obviously always
;; applicable.



(defun derivative (fn)
  (cond ((member fn (list #'linear 'linear))
         #'d-linear)
        ((member fn (list #'rectified-linear 'rectified-linear))
         #'d-rectified-linear)
        ((member fn (list #'logistic 'logistic))
         #'d-logistic)
        ((member fn (list #'softmax 'softmax))
         #'d-logistic)
        (t (error "Don't know the derivative of ~A."
                  fn))))

(defun back-propagation (inputs network output-deriv a z)
  "Back propagate the derivatives of a batch of inputs through the network."
  (let ((nl (network-list network))
        (rev-as (reverse (cons inputs a)))
        (rev-zs (reverse (cons inputs z))))
    ;; back propagate
    (reverse
     (iter (for (th units) :in (reverse nl))
       (for (a-prev a) :on rev-as)
       (for z :in (rest rev-zs))
       (for p-delta = (if (first-iteration-p)
                          output-deriv
                          delta))
       (for delta = (map-ima #'*
                             (unmap-into 'array (remove-biases (m*m p-delta th)))
                             (map-ima (lambda (x y) (map-ima funcall (derivative x) ) (unmap-into 'array z))))
       (collecting
        (map-ima (lambda (x) (/ x (ima-dimension inputs 0)))
                 (unmap-into
                  'array
                  (transpose (m:* (transpose (add-biases a)) p-delta)))))))))

