
(ql:quickload :stefil)

(defpackage :clann-test
  (:use :cl :clann :stefil :iterate :ima)
  (:export))

(in-package :clann-test)

(in-root-suite)

(defsuite :forward-propagation)

(defparameter *rel-tol* 1d-5)
(defparameter *abs-tol* 1d-10)

(defun equalp~ (a b)
  (cond ((ima-p a)
         (equal (ima-dimensions a) (ima-dimensions b))
         (iter
           (for a-el :in-ima a)
           (for b-el :in-ima b)
           (always (equalp~ a-el b-el))))
        ((numberp a)
         (let ((mag (/ (+ (abs a) (abs b))
                       2)))
           (or
            ;; Relative error (ignore errors for the case where there is a
            ;; division by zero)
            (ignore-errors
             (< (/ (abs (- a b)) mag)
                *rel-tol*))
            ;; Absolute error
            (< (abs (- a b))
               *rel-tol*))))
        ((atom a) (equalp a b))))

(deftest %reference-forward-propagation (function solution transpose-solution)
  ;; We will specify the network by hand, something that you would basically
  ;; never do under normal circumstances.
  (let ((network (make-instance
                  'clann:clan-net
                  :network-list `((#2A((1 2)
                                       (2 3)
                                       (4 5)
                                       (6 7))
                                      ,function)))))
    (is (equalp~ (forward-propagation
                  #2A((0.1)
                      (-1.1)
                      (2.1)
                      (3.1)
                      (-4.1))
                  network)
                 solution)))
  (let ((network (make-instance
                  'clann:clan-net
                  :network-list
                  `((#2A((1 2 4 6)
                         (2 3 5 7))
                        ,function)))))
    (is (equalp~ (forward-propagation
                  #2A((0.1 0.2 0.3)
                      (-1.1 -1.2 -1.3)
                      (2.1 2.2 2.3)
                      (3.1 3.2 3.3)
                      (-4.1 -4.2 -4.3))
                  network)
                 transpose-solution))))

;; These are hand checked (or need to be)
(deftest reference-forward-propagation ()
  (%reference-forward-propagation
   'logistic
   '(#2A((0.7685248 0.9088771 0.9890131 0.99877053)
         (0.23147522 0.21416499 0.18242553 0.15446523)
         (0.99451375 0.9997515 0.9999995 1.0)
         (0.9992539 0.9999876 1.0 1.0)
         (7.4602896e-4 3.363199e-5 6.825603e-8 1.3852104e-10)))
   '(#2A((0.97811866 0.9955037)
         (1.0156313e-6 7.5434606e-8)
         (1.0 1.0)
         (1.0 1.0)
         (2.3557732e-22 2.1593257e-27))))
  (%reference-forward-propagation
   'linear
   '(#2A((1.2 2.3 4.5 6.7)
        (-1.2 -1.3000002 -1.5 -1.7000003)
        (5.2 8.299999 14.5 20.699999)
        (7.2 11.299999 19.5 27.699999)
        (-7.2 -10.299999 -16.5 -22.699999)))
   '(#2A((3.8000002 5.4)
         (-13.799999 -16.4)
         (27.8 35.4)
         (39.8 50.4)
         (-49.800003 -61.4))))
  (%reference-forward-propagation
   'rectified-linear
   '(#2A((1.2 2.3 4.5 6.7)
         (0 0 0 0)
         (5.2 8.299999 14.5 20.699999)
         (7.2 11.299999 19.5 27.699999)
         (0 0 0 0)))
   '(#2A((3.8000002 5.4)
         (0 0)
         (27.8 35.4)
         (39.8 50.4)
         (0 0))))
  (%reference-forward-propagation
   'softmax
   '(#2A((3.320117 9.974182 90.01713 812.4057)
         (0.3011942 0.27253175 0.22313017 0.18268347)
         (181.2722 4023.8694 1982759.3 9.770016e8)
         (1339.4305 80821.58 2.9426755e8 1.07141235e12)
         (7.4658595e-4 3.363312e-5 6.8256035e-8 1.3852104e-10)))
   '(#2A((44.701195 221.40643)
         (1.0156323e-6 7.543461e-8)
         (1.1840942e12 2.3660578e15)
         (1.92717e17 7.7346835e21)
         (2.3557732e-22 2.1593257e-27)))))

(deftest %reference-back-propogation (
