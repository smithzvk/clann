
(in-package :smithzv.clann)

;; <<>>=
(defclass clan-net ()
  ((network-list :accessor network-list :initarg :network-list :initform nil)))

;; <<>>=
(defmethod print-object ((network clan-net) str)
  (if *print-readably*
      (print (network-list network) str)
      (format str "#<CLAN-NET #~A ~A>"
              (string-downcase (subseq (format nil "~36R" (sxhash network)) 0 7))
              (let ((nl (network-list network)))
                (cons (- (ima-dimension (caar nl) 1) 1)
                      (mapcar (lambda (x) (ima-dimension (first x) 0)) nl))))))

;; <<>>=
(defun layer (network n)
  (elt (network-list network) n))
;; <<>>=
(defun make-network (layers
                         &key (initialize-weights 0.01d0)
                              (matrix-generation-fn #'generate-lisp-array))
  "Make a neural network with the given layer specification.  LAYERS is a list
with one element per layer \(counting the input and output as layers).  Each
element in LAYERS should be either an integer \(specifying the number of units
in that layer) or a list of the form \(N-UNITS &OPTIONAL UNIT-TYPE).  UNIT-TYPE
determines activation function of the neurons in this layer and can be an symbol
or a list of symbols or length N-UNITS.

Weights are initialized by sampling a double float in the range \(\(-
INITIALIZE-WEIGHTS) INITIALIZE-WEIGHTS) or, if MATRIX-GENERATION-FN is set to a
user function, the user is responsible for this initialization.  Your function
will be passed the number of needed rows and columns and a function that will
return randoms numbers when called."
  (let ((fn (lambda () (- (random (* 2 (float initialize-weights 0d0)))
                     initialize-weights)))
        (n-inputs (first layers)))
    (make-instance
     'clan-net
     :network-list
     (iter (for (n-units . rest-layer) in (mapcar
                                           'alexandria:ensure-list
                                           (rest layers)))
       (for last-n-units previous n-units)
       (destructuring-bind (&key (unit-type 'logistic))
           rest-layer
         (collecting
          (list (funcall matrix-generation-fn n-units
                         (1+ (or last-n-units n-inputs))
                         fn)
                unit-type)))))))

;; <<>>=
(defun generate-matrix (n m fn)
  "Generate a list IMA that is n by m and with elements sampled from fn."
  (iter (for i below n)
    (collecting
     (iter (for j below m)
       (collecting (funcall fn))))))

;; <<>>=
(defun generate-lisp-array (n m fn)
  "Generate a matrix as a lisp-array that is n by m with elements sampled from
fn."
  (ima:unmap-into 'array (generate-matrix n m fn)))
