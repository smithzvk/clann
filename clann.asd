
(asdf:defsystem :clann
  :author "Zach Kost-Smith"
  :license "Lesser Lisp GPL"
  :depends-on (:index-mapped-arrays :iterate :bvecops)
  :serial t
  :components ((:file "package")
               (:file "utils")
               (:file "forward-propagation")
               (:file "back-propagation")
               (:file "train")))
