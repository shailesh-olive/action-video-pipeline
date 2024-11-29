import easyocr
import numpy as np
from CRAFT import CRAFTModel


class JerseyOCR:
    def __init__(self) -> None:
        self.reader = easyocr.Reader(lang_list=["en"])
        self.model = CRAFTModel("weights/", "cuda", use_refiner=True, fp16=True)

    def get_jersey_number(self, image: np.ndarray):
        # use craft text detection model to detect jersey numbers
        polygons = self.model.get_polygons(image)

        pred_num, confidence = 0, 0.0
        if polygons:
            # get bbox of the detected jersey number
            bboxes = np.array(polygons[0]).T

            x_coord = bboxes[0]
            y_coord = bboxes[1]

            # get bbox coord of the jersey image
            x1 = abs(min(x_coord))
            y1 = abs(min(y_coord))
            x2 = abs(max(x_coord))
            y2 = abs(max(y_coord))

            h, w, _ = image.shape

            # add padding only if the x, y value is greater than 5
            x1 = x1 - 5 if x1 >= 5 else x1
            y1 = y1 - 5 if y1 >= 5 else y1

            x2 = max(x_coord) + 5
            y2 = max(y_coord) + 5

            # remove padding of x2 and y2
            # if it is greater than image width and height
            x2 = x2 - 5 if x2 > w else x2
            y2 = y2 - 5 if y2 > h else y2

            jersey_img = image[y1:y2, x1:x2]
            preds = self.reader.readtext(jersey_img, allowlist="0123456789")

            if preds:
                pred_num = preds[0][1]
                confidence = preds[0][2]
                confidence = float(f"{confidence:.2f}")
                if confidence > 0.7 and len(pred_num) < 3:
                    return pred_num, confidence
                else:
                    return 0, 0.0

        return pred_num, confidence
