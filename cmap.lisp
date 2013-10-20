
(defpackage :clann.cmap
  (:use :cl :cffi)
  (:export))

(in-package :clann.cmap)

(define-foreign-library cmap
  (t (:default "./libcmap")))

(use-foreign-library cmap)

(defcfun cmap :int
  (arr (:pointer :double))
  (n :int)
  (m :int)
  (function func))

(defcfun cmap-fn :int
  (arr (:pointer :double))
  (n :int)
  (m :int)
  (function func))

