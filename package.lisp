
(in-package :cl-user)

(defpackage :smithzv.clann
  (:use :cl :index-mapped-arrays :iterate)
  (:export
   #:layer
   #:clan-net
   #:generate-lisp-array
   #:make-network
   #:network-list))
