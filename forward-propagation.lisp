
(in-package :clann)

(defun logistic (x) (/ (+ 1 (exp (- x)))))

(defun linear (x) x)

(defun rectified-linear (x) (if (< x 0) 0 x))

;; This has issues when it comes to running on a gpu...
(defun stocastic-binary (fn)
  (lambda (x)
    (if (< (random 1d0) (funcall fn x)) 0 1)))

;; @\section{A Reference Implementation}

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

(defun softmax (x)
  (exp x))

(defun funcall-columns (fn m &optional ret)
  "Multiply matrices together.  This is just a naive way to do this for
verification purposes.  Use RET as the output if provided, otherwise allocate a
new matrix."
  (unless (= (ima-dimension fn 0)
             (ima-dimension m 0))
    (error "Incompatible dimensions of matrices: ~A and ~A"
           (ima-dimensions fn)
           (ima-dimensions m)))
  (let ((ret* (or ret (make-array (ima-dimensions m)))))
    ;; Summing order swapped, should be better for cache...
    (iter (for i :below (ima-dimension m 0))
      (iter (for j :below (ima-dimension m 1))
        (setf (imref ret* i j) (funcall (imref fn i) (imref m i j)))))
    ret*))

(defun apply-unit-functions (z unit-functions)
  "Map functions in unit-functions to each column in z."
  (let ((unit-functions (if (atom unit-functions)
                            (make-list (ima-dimension z 1)
                                       :initial-element unit-functions)
                            unit-functions)))
    (let ((ret (make-ima-like z))
          (softmaxs '())
          (softmax-total (map-ima (constantly 0) (column-vector z 0))))
      (iter (for fn :in-sequence unit-functions :with-index i)
        (setf (column-vector ret i)
              (let ((val (map-ima fn (column-vector z i))))
                 (when (eql fn #'softmax)
                   (push i softmaxs)
                   (map-ima #'+ softmax-total val))
                 val)))
      (iter (for i :in softmaxs)
        (setf (column-vector ret i)
              (map-ima #'/ (column-vector ret i) softmax-total)))
      ret)))

(defun forward-propagation (inputs network)
  "Forward propagate a batch of inputs through the network."
  (let ((nl (network-list network))
        (z nil)
        (a inputs))
    (iter (for (th units) :in nl)
      (setf z (m*m (add-biases a) (ima:transpose th))
            a (apply-unit-functions z units))
            ;; a (ima:map-ima 'logistic (unmap-into 'array z)))
      (collect a :into as)
      (collect z :into zs)
      (finally (return (values as zs))))))

(defun predict (input network)
  (let* ((%input (if (= (length (ima-dimensions input)) 1)
                     (unmap (ima:group-imas (list input) 0))
                     input))
         (results (forward-propagation %input network)))
    (if (eql %input input)
        (first (last results))
        (row-vector (first (last results)) 0))))


;; (ql:quickload :stefil)

;; (stefil:deftest train-identity ()
;;   (let ((net (make-network '(1 1))))
;;     (gradient-descent '((1) (0)) net '((1) (0)))))

;; (stefil:deftest train-not ()
;;   (let ((net (make-network '(1 1))))
;;     (gradient-descent '((1) (0)) net '((0) (1)))))

;; (stefil:deftest train-constant ()
;;   (let ((net (make-network '(1 1))))
;;     (gradient-descent '((1) (0)) net '((1) (1)))
;;     (gradient-descent '((1) (0)) net '((0) (0)))))

;; @\subsection{Validation}

;; We can verify that this system works with simple unitary and boolean
;; functions.

;; (defparameter *or-net* (make-network '(2 1)))

;; (gradient-descent '((1 1) (0 1) (1 0) (0 0))
;;                   *or-net*
;;                   '((1) (1) (1) (0))
;;                   :max-steps 100000)

;; ((#2A((-1.6151030664969923d0 3.0611447275121697d0 3.0622881749687316d0))
;;   LOGISTIC))

;; (mapcar (lambda (x) (predict x *or-net*))
;;         '((0 0) (1 0) (0 1) (1 1)))

;; (#(0.18761492830472365d0) #(0.9271971950689464d0) #(0.9272073152295347d0)
;;  #(0.9985784094744862d0))

;; @\subsubsection{Balance}

;; When we attempt to train the OR function, we find an interesting result, the
;; network tends towards making a function that will always return 1 for any input.

;; This is actually a fairly accurate thing for the network to do.  Most of the
;; time "1" is the correct answer.  To first order, the OR function is {/em
;; (constantly 1)}.  If you train long enough, the correct value will be found.

;; Another way to alleviate this problem is to give a balanced data set.  You
;; can bootstrap sample equally from the two distributions (those that result in
;; 0 and those that result in 1).  This will remove the tendency for the network
;; to use the knowledge that one category is much more likely than the other.

;; @\subsubsection{XOR}

;; The XOR function has unique problems when it comes to training a neural
;; network.  This is related to some kind of symmetry of the problem which
;; cancels the derivatives when several training samples are averaged over.  I
;; think that this symmetry exists near the zero weight state of the network.
;; This means that we can get around it if you break the symmetry by, say,
;; initializing the network with larger weights, or by starting to train the
;; network using a related function like OR, then retraining it for XOR after
;; the weights have increased a bit.

;; @\subsubsubsection{Learned Features}

;; The XOR function requires a hidden layer to accurately model the behavior.
;; This is often not easily determined, but in this simple case we can see this
;; by trying every possible two input/one output network.  This hidden layer can
;; be thought of as a layer of learned features.  Instead of the features $x$
;; and $y$, the network learns two new features.  We can examine the connection
;; weights to find out what those features are.  Here is an example of what you
;; might get if you train a 2-2-1 network (I believe this is the minimal network
;; for this function):

;; ((#2A((-8.37302024737629d0 4.9060393726892855d0 4.906046195853849d0)
;;       (-2.487132780883222d0 5.032253965356175d0 5.032262616061907d0))
;;   LOGISTIC)
;;  (#2A((-3.369310643066896d0 -11.578578538173351d0 7.951780794421816d0))
;;   LOGISTIC))

;; @Looking at this, we can see that the output's dependence on the hidden layer
;; is along the lines of "the output is 0 if both units are 0, 0 if unit one is
;; 1 regardless of what unit two does, and 1 if unit one is 0 and unit two is
;; 1", or in the language of Lisp:

;; (if (= unit-one 1)
;;     0
;;     (if (= unit-two 1)
;;         1
;;         0))

;; @A bit of thinking tells us that this behavior fits if unit-one's value
;; designates the input was (1,1) and unit-two's value designates that the input
;; was not (0,0).  Looking at the weights in the first matrix confirms this
;; interpretation.  This is a perfectly good representation for us to talk
;; about, but it is absolutely essential for this neural network.  What is
;; happening in this example is that the network is learning a representation
;; that it can use to think about the XOR problem.  The first weight matrix
;; represents that translation from the input representation into the internal
;; representation.  Then the network uses this internal representation to
;; compute the XOR function.

;; Also, note that the weights are quite large in this case.  This can be a
;; problem and we will deal with it the section {\em Regularization Methods}.

;; @\subsubsubsection{Network Capacity, Bias, and Variance}

;; The fact that a hidden layer is necessary brings up the idea of {\em network
;; capacity}, or the complexity of the function that a network can model.  If
;; the capacity of the network is too low for the function you are modeling, the
;; network will have high {\em bias}.  Networks with high capacity are
;; susceptible to high {\em variance}.  Much of the work regarding neural
;; networks (and all of machine learning) is limiting the capacity of the
;; network while allowing it to be large enough to properly model the function.




;; ;; Version with destructive updates

;; ;; This is just a mock-up of how this should behave, but it is useful as a
;; ;; reference implementation of the low level code.

;; (defun forward-propagation* (inputs network
;;                              &optional (batch-size (ima-dimension inputs 0)))
;;   "Forward propagate a batch of inputs through the network."
;;   (let* ((nl (network-list network))
;;          (as (cons
;;               (unmap-into
;;                'array
;;                (append-imas
;;                 (list (make-list batch-size
;;                                  :initial-element '(1))
;;                       (ima:submatrix inputs 0 0 batch-size (ima-dimension inputs 1)))
;;                 1))
;;               (mapcar (lambda (x) (make-array (list batch-size
;;                                                (+ 1 (ima-dimension (first x) 0)))
;;                                          :initial-element 1d0))
;;                       nl))))
;;     (iter
;;       (for (th units) :in nl)
;;       (for (a-prev a) :on as)
;;       (setf (submatrix a 0 1 batch-size (- (ima-dimension th 1) 1))
;;             (bm:* a-prev (ima:transpose th)))
;;       (setf (submatrix a 0 1 batch-size (- (ima-dimension th 1) 1))
;;             (ima:map-ima
;;              'logistic
;;              (unmap-into
;;               'array
;;               (submatrix a 0 1 batch-size (- (ima-dimension th 1) 1))))))
;;     as))

;; ;; Using destructive updates really pays off in the lower level implementations
;; ;; as it requires fewer memory accesses across FFI barriers and allows for less
;; ;; memory copying across the barriers (as well as less memory consing, to boot).
;; ;; The goal here is to have all computation happen in C and, if possible, inside
;; ;; performance tuned libraries like CUBLAS, APPML, or Atlas.

;; ;; Code for running in BLAS (Atlas or some other BLAS)

;; (defcfun map-c-array :void
;;   (units (:pointer :int))
;;   (z (:pointer :double))
;;   (n :int) (m :int) (lda :int))

;; (defun blas-forward-propagation (inputs network
;;                                  &optional (batch-size (ima-dimension inputs 0)))
;;   "Forward propagate a series of batches of inputs through the network.  This
;; works destructively \(for efficiency reasons)."
;;   (let ((nl (network-list network))
;;         (z* nil)
;;         (a* (list inputs)))
;;     (iter
;;       (for (th units) :in nl)
;;       (for z :in (unit-values network))
;;       (cblas-cffi-bindings:cblas-dgemm
;;        :row-major :cblas-no-trans :cblas-no-trans
;;        (ima-dimension a 0)
;;        (ima-dimension b 1)
;;        (ima-dimension b 0)
;;        1d0
;;        (ima-c-array:pointer-of a)
;;        (ima-dimension a 1)
;;        (ima-c-array:pointer-of b)
;;        (ima-dimension b 1)
;;        0d0
;;        (ima-c-array:pointer-of c)
;;        (ima-dimension c 1))
;;       (map-c-array (ima-c-array:pointer-of unit-type)
;;                    (ima-c-array:pointer-of c)
;;                    (ima-dimension c 0)
;;                    (ima-dimension c 1)
;;                    (ima-dimension c 1)))))

;; ;; Code for running using the GPU (cublas and specialized code)

;; (defcfun cuda-map-array :void
;;   (fn (:pointer :int))
;;   (z (:pointer (:pointer :double)))
;;   (n :int) (m :int) (lda :int))

;; (defun cublas-forward-propagation (inputs network
;;                                    &optional (batch-size (ima-dimension inputs 0)))
;;   "Forward propagate a series of batches of inputs through the network.  This
;; works destructively \(for efficiency reasons)."
;;   (let ((nl (network-list network))
;;         (z* nil)
;;         (a* (list inputs)))
;;     (iter
;;       (for (th units) :in nl)
;;       (for z :in (unit-values network))
;;       (cublas-bindings:cublassgemm
;;        :row-major :cblas-no-trans :cblas-no-trans
;;        (ima-dimension a 0)
;;        (ima-dimension b 1)
;;        (ima-dimension b 0)
;;        1d0
;;        (ima-c-array:pointer-of a)
;;        (ima-dimension a 1)
;;        (ima-c-array:pointer-of b)
;;        (ima-dimension b 1)
;;        0d0
;;        (ima-c-array:pointer-of c)
;;        (ima-dimension c 1))
;;       (map-c-array (ima-c-array:pointer-of unit-type)
;;                    (ima-c-array:pointer-of c)
;;                    (ima-dimension c 0)
;;                    (ima-dimension c 1)
;;                    (ima-dimension c 1)))))

;; ;; Code for running using APPML

;; (defcfun opencl-map-array :void
;;   (fn (:pointer :int))
;;   (z (:pointer (:pointer :double)))
;;   (n :int) (m :int) (lda :int))

;; (defun appml-forward-propagation (inputs network
;;                                   &optional (batch-size (ima-dimension inputs 0)))
;;   "Forward propagate a series of batches of inputs through the network.  This
;; works destructively \(for efficiency reasons)."
;;   (let ((nl (network-list network))
;;         (z* nil)
;;         (a* (list inputs)))
;;     (iter
;;       (for (th units) :in nl)
;;       (for z :in (unit-values network))
;;       (cublas-bindings:cublassgemm
;;        :row-major :cblas-no-trans :cblas-no-trans
;;        (ima-dimension a 0)
;;        (ima-dimension b 1)
;;        (ima-dimension b 0)
;;        1d0
;;        (ima-c-array:pointer-of a)
;;        (ima-dimension a 1)
;;        (ima-c-array:pointer-of b)
;;        (ima-dimension b 1)
;;        0d0
;;        (ima-c-array:pointer-of c)
;;        (ima-dimension c 1))
;;       (map-c-array (ima-c-array:pointer-of unit-type)
;;                    (ima-c-array:pointer-of c)
;;                    (ima-dimension c 0)
;;                    (ima-dimension c 1)
;;                    (ima-dimension c 1)))))
