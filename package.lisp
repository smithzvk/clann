
(in-package :cl-user)

(defpackage :clann
  (:use :cl :index-mapped-arrays :iterate)
  (:export
   #:layer
   #:clan-net
   #:generate-lisp-array
   #:make-network
   #:network-list
   #:forward-propagation
   #:logistic
   #:linear
   #:rectified-linear
   #:softmax))

;; @\section{System Overview}

;; CLANN is a Common Lisp system for modeling Artificial Neural Networks (ANNs).
;; CLANN provides efficient routines to train ANNs based on observations and to
;; use ANNs to predict the behavior of new observations.  CLANN also aims to
;; provide a simple interface for training ANN.  The goal is to provide a
;; facility for the correct thing to happen by default, but still be flexible to
;; provide users with a platform that will adapt to their needs.  CLANN aims to
;; provide the following features:

;; \begin{itemize}

;; \item A fast feed-forward and back-propagation implementation

;; \item Constraints on activities and gradients

;; \item Proper handling of datasets (training, cross-validation, test)

;; \end{itemize}

;; CLANN is designed around matrix multiplication.  To CLANN, any ANN is seen as
;; a set weight matrices that determine how inputs are mapped to hidden layers,
;; hidden layers to other hidden layers, and finally how the final layer is
;; mapped to outputs.  This trivially allows for the modeling of feed forward
;; networks.  It also allows for recurrent networks as any recurrent network can
;; be modeled as a feed-forward network via unrolling the recurrence in time and
;; applying some constraints to the gradients on the weights.


;; This is the ugly system I have right now:

;; Each layer can have any number of units and each unit can be one either a
;; linear, rectified-linear, logistic, or softmax.  Softmax are special units
;; that actually communicate with other neurons within the softmax to model a
;; the ${\rm max}$ function over a probability distribution.

;; You cannot specify other unit types than those that are baked in.  You cannot
;; use cost functions other than the blessed ones included in the system.  This
;; seems pretty limiting.  Why?  It will certainly be this limited when you get
;; down to the C and CUDA implemented systems.
