import numpy as np
import cv2

class OpticalFlow:

    def __init__(self,sequence_length=2, normalize=False):
        self.win = cv2.namedWindow("webcam stream")
        self.vc = cv2.VideoCapture(0)
        self.window_height = 480
        self.window_width = 640
        self.discretization = 0.0625 
        self.step = int(self.window_width * self.discretization)
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)

        if self.vc.isOpened():
            rval, frame = self.vc.read()
            self.previousImages = [frame for _ in range(sequence_length)]
        else: 
            rval = False

        while rval:
            rval, frame = self.vc.read()
            frame = cv2.flip(frame, 1)
            self.shiftImageSequence(frame)
            imageSequence = self.previousImages.copy()
            self.optical_flow_matrix = np.ndarray((int(self.window_height // self.step),int(1 / self.discretization), 2))
            for i in range(sequence_length-1):
                # derivate wrt time, x and y
                It = cv2.subtract(imageSequence[i], imageSequence[i+1], dtype=cv2.CV_32F)
                Ix = cv2.Sobel(imageSequence[i], cv2.CV_32F, 1, 0)
                Iy = cv2.Sobel(imageSequence[i], cv2.CV_32F, 0, 1)
                for ixj, j in enumerate(range(0, self.window_width, self.step)):
                    for ixk, k in enumerate(range(0, self.window_height, self.step)):
                        # grab the specific area of the optical flow to be computed
                        iteration_It = It[k:k + self.step, j:j + self.step]
                        iteration_Ix = Ix[k:k + self.step, j:j + self.step]
                        iteration_Iy = Iy[k:k + self.step, j:j + self.step]

                        # flatten them to use them in the motion constraint equation
                        IxColumVector = np.array([np.resize(iteration_Ix, iteration_Ix.size)]).T
                        IyColumVector = np.array([np.resize(iteration_Iy, iteration_Iy.size)]).T
                        ItColumVector = np.array([np.resize(iteration_It, iteration_It.size)]).T

                        # matrix of x and y derivatives
                        mat = np.hstack((IxColumVector, IyColumVector))
                        # compute moore penrose pseudo inverse
                        pseudo_inverse = np.linalg.pinv(mat)
                        # solve system of equations ([Ix, Iy] . [u, v] = It)
                        self.optical_flow_matrix[ixk, ixj] =  pseudo_inverse.dot(ItColumVector).flatten()

            half_step = self.step // 2

            for ixj, j in enumerate(range(0, self.window_width, self.step)):
                for ixk, k in enumerate(range(0, self.window_height, self.step)):
                    optical_flow_vector = self.optical_flow_matrix[ixk,ixj].flatten()
                    norm = np.linalg.norm(optical_flow_vector)
                    if normalize:
                        optical_flow_vector /= norm
                    if norm > 0.25:
                        optical_flow_vector *= 10
                    optical_flow_vector = np.int16(optical_flow_vector)
                    cv2.arrowedLine(frame, (j+half_step, k+half_step), (j+half_step+optical_flow_vector[0], k+half_step+optical_flow_vector[1]), (255,255,255), 1, tipLength=0.5)

            cv2.imshow("webcam stream", frame)
            key = cv2.waitKey(20)
            if key == 27:
                cv2.destroyAllWindows()
                self.vc.release()
                break

    def shiftImageSequence(self, newImage):
        self.previousImages = [i for i in self.previousImages[1:] + [newImage]]
            

if __name__ == '__main__':
    OpticalFlow(2, True)
