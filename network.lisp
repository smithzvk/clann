
(in-package :clann)

;; @A network is a list of index-mapped-arrays.  These IMAs represents the
;; weight matrices (including the bias row).  <<Make-network>> will generate a
;; network with pre-initialized weights (randomly for now, but in the future I
;; am looking towards other methods).  Because neural networks can get large and
;; lists are printed out on the REPL, <<make-network>> will actually return a
;; class instance that contains the network list to allow for the programmatic
;; control of the printer.

;; <<>>=
(defclass clan-net ()
  ((network-list :accessor network-list :initarg :network-list :initform nil)
   (unit-values :accessor unit-values :initarg :unit-values :initform nil)))

;; @The network will be printed as...

;; #<CLAN-NET #<unique-id> (<#-inputs> [<#-units-in-hidden-layer>]* <#-outputs>)>

;; ...which will hopefully be enough to distinguish different networks easily.
;; The network list is printed if <*print-readably*> is non-nil.  You can access
;; the network list programmatically via the <<network-list>> function, and
;; individual layers can be accessed via the <<layer>> function.

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

;; @A neural network (as it is modeled in CLANN) is nothing more that a set of
;; matrices that define how the activity of one layer maps to the inputs of the
;; next layer {\em and} how those inputs map to activities in that next layer.
;; This means that for a neural network with $n$ layers (1 input layer, one
;; output layer, and $n-2$ hidden layers) there will be $n-1$ matrices mapping
;; between the layers.  Thus the length of the network list, is $n-1$.  You
;; specify the number of units in each layer by specifying a list of integers.
;; For instance, to make a network that has 5 inputs, a hidden layer with 7
;; units, a hidden layer with 2 units, and 1 output, you would do something like
;; this:

;; (make-network '(5 7 2 1))

;; The unit type is also part of the mapping from activity in one layer to
;; activity in the next layer.  In neural networks, logistic units are by far
;; the most common (and the default in CLANN), but you can set the unit type to
;; other options (discussed later) by specifying it with the number of units.
;; This can be done by using (instead of an integer) a list of the form
;; <(n-units unit-type)>.  In this list <unit-type> is either a symbol denoting
;; a type or unit which will set all to units in that layer, or a list that sets
;; each unit independently.

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
     (iter (for (n-units . rest-layer) :in (mapcar
                                            'alexandria:ensure-list
                                            (rest layers)))
       (for last-n-units :previous n-units)
       (destructuring-bind (&key (unit-type 'logistic))
           rest-layer
         (let ((unit-type (if (listp unit-type)
                              unit-type
                              (make-list (or last-n-units n-inputs)
                                         :initial-element unit-type))))
           (collecting
            (list (funcall matrix-generation-fn n-units
                           (1+ (or last-n-units n-inputs))
                           fn)
                  unit-type)))))
     :unit-values
     (iter (for n-units :in (mapcar 'alexandria:ensure-car (rest layers)))
       (collecting
        (make-array n-units :initial-element 0d0))))))

;; @The function <<generate-lisp-array>> will create a new matrix (represented
;; as a Lisp array) whose elements are sampled from the function <fn>, which
;; randomly returns values between $(-{\rm initialize-weights},{\rm
;; initialize-weights})$.  The point of setting random values is that it splits
;; the symmetry in the learning algorithm.  Without splitting that symmetry, the
;; network cannot learn effectively.

;; Actually building the matrix is done by the (unexported) helper function
;; <<generate-matrix>>.  You may specify a matrix generation function of your
;; own via <matrix-generation-fn> in which case you are responsible for
;; returning an IMA that is properly initialized for the network.  Your function
;; will be passed the number of rows and columns and a function that will
;; generate random numbers if called.

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
