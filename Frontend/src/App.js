import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Row, Col, Button } from "react-bootstrap";
import pic from "./images/picture.svg";
import axios from "axios";

function App() {
  const hiddenFileInput = React.useRef(null);
  const [image, setImage] = React.useState(pic);
  const [flower, setFlower] = React.useState(null);
  const [resFlower, setResFlower] = React.useState("")

  const handleClick = (event) => {
    event.preventDefault();
    hiddenFileInput.current.click();
  };

  const handleChange = (event) => {
    event.preventDefault();
    setResFlower("Click upload dude!!!")
    setImage(URL.createObjectURL(event.target.files[0]));
    setFlower(event.target.files[0]);
  };

  const uploadImage = async () => {
    if (flower) {
      setResFlower("Loading...")
      var formData = new FormData();
      formData.append("file", flower);
       await axios
        .post("http://35.223.117.5:8000/prediction", formData, {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            'Access-Control-Allow-Origin': '*'
          },
        })
        .then(function (res) {
          console.log("SUCCESS!!");
          console.log(res.data.prediction.flowers);
          setResFlower(res.data.prediction.flowers)

        })
        .catch(function () {
          console.log("FAILURE!!");
           setResFlower("Something went wrong with server!!!")
        });
    
    }
  };
  return (
    <>
      <div
        style={{
          width: "100%",
          height: "100vh",
          background: "#dbebfc",
          padding: "20px",
        }}
      >
        <Row
          sm
          className="h-20 w-100 d-flex justify-content-center align-content-center"
          style={{ margin: 0 }}
        >
          <Col
            sm
            className="w-100 d-flex justify-content-center align-content-center"
          >
            <h1>FLOWERs API</h1>
          </Col>
        </Row>
        <div
          className="h-80 w-100 d-flex justify-content-center align-content-center"
          style={{ margin: 0 }}
        >
          <Row
            sm
            className="d-flex justify-content-center align-content-center"
            style={{ width: "400px", height: "550px", padding: "20px" }}
          >
            <Col
              sm
              style={{
                width: "100%",
                height: "100%",
                borderRadius: "30px",
                backgroundColor: "white",
                padding: "10px",
                boxShadow: "2px 3px 5px 0px rgba(158,158,158,1)",
              }}
            >
              <Row
                sm
                style={{
                  width: "100%",
                  height: "70%",
                  margin: 0,
                  paddingBottom: 20,
                }}
              >
                <Col
                  sm
                  className="d-flex justify-content-center align-content-center"
                  style={{ width: "100%", height: "100%", margin: 0 }}
                >
                  <img
                    src={image}
                    style={{ maxHeight: "100%", maxWidth: "100%" }}
                    alt="pic"
                  />
                </Col>
              </Row>
                            <Row
                sm
                className="w-100 h-10"
                style={{ margin: 0, paddingBottom: 20 }}
              >
                <Col sm className="w-100 h-100">
                <h5>Your flower :{"  "+resFlower}</h5>
                </Col>
              </Row>
              <Row
                sm
                className="w-100 h-10"
                style={{ margin: 0, paddingBottom: 20 }}
              >
                <Col sm className="w-100 h-100">
                  <Button
                    onClick={handleClick}
                    variant="primary"
                    className="w-100 h-100"
                  >
                    CHOOSE YOUR FLOWER
                  </Button>
                  <input
                    onChange={handleChange}
                    type="file"
                    ref={hiddenFileInput}
                    style={{ display: "none" }}
                  />
                </Col>
              </Row>
              <Row
                sm
                className="w-100 h-10"
                style={{ margin: 0, paddingBottom: 20 }}
              >
                <Col sm className="w-100 h-100">
                  <Button
                    onClick={uploadImage}
                    variant="success"
                    className="w-100 h-100"
                  >
                    UPLOAD
                  </Button>
                </Col>
              </Row>
            </Col>
          </Row>
        </div>
      </div>
    </>
  );
}

export default App;
