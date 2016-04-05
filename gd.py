import numpy as np
import sklearn
from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats

class GD:
    def __init__(self, alpha, x, y, ep=0.01, max_iter=10000):
        self.alpha = alpha;
        self.x = x;
        self.y = y;
        self.ep = ep;
        self.max_iter = max_iter;

    def start(self):
        converged = False
        iter = 0
        m = self.x.shape[0] # number of samples

        # initial theta
        t0 = np.random.random(1)
        t1 = np.random.random(1)

        # total error, J(theta)
        J = sum([(t0*self.x[i][0] + t1*self.x[i][1] - self.y[i]) ** 2 for i in range(m)])

        # print "Staring gradient descent......"
        # Iterate Loop
        while not converged:
            # for each training sample, compute the gradient (d/d_theta j(theta))
            grad0 = 1.0/m * sum([(t0*self.x[i][0] + t1*self.x[i][1] - self.y[i])*self.x[i][0] for i in range(m)]) 
            grad1 = 1.0/m * sum([(t0*self.x[i][0] + t1*self.x[i][1] - self.y[i])*self.x[i][1] for i in range(m)])

            # update the theta_temp
            temp0 = t0 - self.alpha * grad0
            temp1 = t1 - self.alpha * grad1

            # update theta
            t0 = temp0
            t1 = temp1

            # mean squared error
            e = sum( [ (t0*self.x[i][0] + t1*self.x[i][1] - self.y[i]) ** 2 for i in range(m)] ) 
            
            # if iter % 50 == 0:
            #     print "Iter: %d, Error: %f" % (iter, abs(J-e)[0])

            if abs(J-e) <= self.ep:
                # print 'Converged, iterations: ', iter, '!!!'
                converged = True
        
            J = e   # update error 
            iter += 1  # update iter
            if iter == self.max_iter:
                # print 'Max interactions exceeded!'
                converged = True
        return t0, t1

# if __name__ == '__main__':
#     x = np.array([[0.434,0.2], [0.123,0.423], [0.343,0.898], [0.123,0.34]])
#     y = np.array([1, 0, 0, 0])
#     print 'x.shape = %s y.shape = %s' %(x.shape, y.shape)
#
#     alpha = 0.001 # learning rate
#     ep = 0.000001 # convergence criteria
#
#     gd = GD(alpha, x, y, ep)
#     # call gredient decent, and get intercept(=theta0) and slope(=theta1)
#     print x, y
#     theta0, theta1 = gd.start()
#     print ('theta0 = %s theta1 = %s') %(theta0, theta1)
