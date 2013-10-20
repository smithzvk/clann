
#include <math.h>

typedef enum 
{
   LOGISTIC,
   RECTIFIED_LINEAR,
   BINARY
} func;

inline double logistic(double val)
{
   return 1/(1+exp(-val));
}

inline double rectified_linear(double val)
{
   if (val < 0)
      return 0;
   else
      return val;
}

inline double binary(double val)
{
   if (val < 0)
      return 0;
   else
      return 1;
}

int cmap (double *arr, int n, int m, func function)
{
   int i,j;
   for (i = 0; i < n; i++)
   {
      for (j = 0; j < m; j++)
      {
         double value, result;

         value = arr[i*m + j];
         switch (function)
         {
            case LOGISTIC:
               result = logistic(value);
               break;
            case RECTIFIED_LINEAR:
               result = rectified_linear(value);
               break;
            case BINARY:
               result = binary(value);
               break;
            default:
               return -1;
         }
            
         arr[i*m + j] = result;
      }
   }
   return 0;
}

int cmap_fn (double *arr, int n, int m, func *function)
{
   int i,j;
   for (i = 0; i < n; i++)
   {
      for (j = 0; j < m; j++)
      {
         double value, result;

         value = arr[i*m + j];
         switch (function[j])
         {
            case LOGISTIC:
               result = logistic(value);
               break;
            case RECTIFIED_LINEAR:
               result = rectified_linear(value);
               break;
            case BINARY:
               result = binary(value);
               break;
            otherwise:
               return -1;
         }
            
         arr[i*m + j] = result;
      }
   }
   return 0;
}
