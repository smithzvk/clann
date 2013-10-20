
(in-package :clann)

(defun m*m (m1 m2 &optional ret)
  "Multiply matrices together.  This is just a naive way to do this for
verification purposes.  Use RET as the output if provided, otherwise allocate a
new matrix."
  (unless (= (ima-dimension m1 1)
             (ima-dimension m2 0))
    (error "Incompatible dimensions of matrices: ~A and ~A"
           (ima-dimensions m1)
           (ima-dimensions m2)))
  (let ((ret* (or ret
                  (make-array (list (ima-dimension m1 0)
                                    (ima-dimension m2 1))
                              :initial-element 0))))
    (unless (eql ret* ret)
      (iter (for i :below (ima-dimension m1 0))
        (iter (for j :below (ima-dimension m2 1))
          (incf (imref ret* i j) 0))))
    ;; Summing order swapped, should be better for cache...
    (iter (for i :below (ima-dimension m1 0))
      (iter (for k :below (ima-dimension m1 1))
        (iter (for j :below (ima-dimension m2 1))
          (incf (imref ret* i j) (* (imref m1 i k) (imref m2 k j))))))
    ret*))

(defun bm*m (m1 m2 &optional ret)
  "Multiply matrices together using BLAS.  Use RET as the output matrix if
provided, otherwise allocate a new matrix."
  )
