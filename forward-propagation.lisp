
(in-package :clann)

(defun logistic (x) (/ (+ 1 (exp (- x)))))

;; A reference implementation

;; This is a slow but almost assuredly correct implementation of the forward and
;; back propagation procedures.  The <<forward-propagation>> function will
;; perform the necessary calculations to determine the activities that result
;; from the given inputs.  The <<predict>> function uses forward propagation to
;; perform a prediction based on the input.

;; The helper functions <<add-biases>> and <<remove-biases>> abstract away the
;; process of adding and removing the bias activities, which are uniformly 1.

(defun add-biases (arr)
  (unmap-into
   'array
   (append-imas
    (list (make-list (ima-dimension arr 0)
                     :initial-element '(1))
          arr)
    1)))

(defun remove-biases (arr)
  (submatrix arr 0 1))

(defun forward-propagation (inputs network)
  "Forward propagate a batch of inputs through the network."
  (let ((nl (network-list network))
        (z nil)
        (a inputs))
    (iter (for (th units) :in nl)
      (setf z (bm:* (add-biases a) (ima:transpose th))
            a (ima:map-ima 'logistic (unmap-into 'array z)))
      (collect a))))

(defun predict (input network)
  (let* ((%input (if (= (length (ima-dimensions input)) 1)
                     (unmap (ima:group-imas (list input) 0))
                     input))
         (results (forward-propagation %input network)))
    (if (eql %input input)
        (first (last results))
        (row-vector (first (last results)) 0))))
