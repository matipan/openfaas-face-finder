package function

import (
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"net/http"
	"strings"

	"gocv.io/x/gocv"
)

// Handle receives a URL in the request
// that points to an image, downloads the image
// run a deep neural network to detect all the faces
// modifies the image and returns it.
func Handle(req []byte) string {
	net := gocv.ReadNet("./models/model.caffemodel", "./models/deploy.prototxt")
	if net.Empty() {
		return fmt.Sprint("Error reading network model")
	}
	defer net.Close()
	net.SetPreferableBackend(gocv.NetBackendDefault)
	net.SetPreferableTarget(gocv.NetTargetCPU)

	var (
		ratio   = 1.0
		mean    = gocv.NewScalar(104, 177, 123, 0)
		swapRGB = false

		ft string
		b  []byte
	)

	// Download the image and parse it's content type. The only
	// supported content types are `jpg-jpeg and png`.
	{
		res, err := http.Get(string(req))
		if err != nil {
			return fmt.Sprintf("error getting image: %s", err)
		}
		defer res.Body.Close()

		ft = strings.TrimPrefix(res.Header.Get("Content-Type"), "image/")
		if ft != "jpeg" && ft != "jpg" && ft != "png" {
			return fmt.Sprintf("image is of type %s but the only supported types are: jpg, png", ft)
		}

		b, err = ioutil.ReadAll(res.Body)
		if err != nil {
			return fmt.Sprintf("error reading the image: %s", err)
		}
	}

	// Decode the bytes into an image mat.
	img, err := gocv.IMDecode(b, gocv.IMReadColor)
	if err != nil {
		return fmt.Sprintf("error while parsing image: %s", err)
	}
	defer img.Close()

	// Run the image through the neural network and draw
	// the detections on the final image.
	{
		blob := gocv.BlobFromImage(img, ratio, image.Pt(300, 300), mean, swapRGB, false)
		net.SetInput(blob, "")
		prob := net.Forward("")
		detect(&img, prob)
		defer blob.Close()
		defer prob.Close()
	}

	// Encode the image into an array of bytes that can be returned
	// to the user.
	b, err = gocv.IMEncode(gocv.FileExt("."+ft), img)
	if err != nil {
		return fmt.Sprintf("error while encoding resulting image: %s", err)
	}

	return string(b)
}

func detect(frame *gocv.Mat, results gocv.Mat) {
	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		if confidence > 0.3 {
			right := int(results.GetFloatAt(0, i+5) * float32(frame.Cols()))
			bottom := int(results.GetFloatAt(0, i+6) * float32(frame.Rows()))
			left := int(results.GetFloatAt(0, i+3) * float32(frame.Cols()))
			top := int(results.GetFloatAt(0, i+4) * float32(frame.Rows()))
			gocv.Rectangle(frame, image.Rect(left, top, right, bottom), color.RGBA{0, 255, 0, 0}, 2)
		}
	}
}
