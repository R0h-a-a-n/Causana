package main

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	router := gin.Default()

	router.POST("/run-job", func(c *gin.Context) {
		method := c.PostForm("method")
		lags := c.PostForm("lags")
		window := c.PostForm("window")

		fileHeader, err := c.FormFile("file")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "file not found: " + err.Error()})
			return
		}

		file, err := fileHeader.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "could not open uploaded file"})
			return
		}
		defer file.Close()
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)

		part, err := writer.CreateFormFile("file", fileHeader.Filename)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "could not create form file"})
			return
		}
		io.Copy(part, file)

		_ = writer.WriteField("method", method)
		_ = writer.WriteField("lags", lags)
		_ = writer.WriteField("window", window)

		writer.Close()
		resp, err := http.Post("http://localhost:8000/analyze", writer.FormDataContentType(), &buf)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "engine connection failed: " + err.Error()})
			return
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read engine response"})
			return
		}
		c.Data(resp.StatusCode, "application/json", body)
	})

	router.Run(":8081")
}
