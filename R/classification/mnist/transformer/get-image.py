import base64
def get_image(data):
	data=data.encode()
	with open("image.png", "wb") as fh:
		fh.write(base64.decodebytes(data))
