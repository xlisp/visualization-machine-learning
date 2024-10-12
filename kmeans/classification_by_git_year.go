package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	rootDir := "./"
	err := filepath.Walk(rootDir, processFile)
	if err != nil {
		log.Fatal(err)
	}
}

func processFile(path string, info os.FileInfo, err error) error {
	if err != nil {
		return err
	}

	if !info.IsDir() && strings.HasSuffix(info.Name(), ".md") {
		year, err := getGitCommitYear(path)
		if err != nil {
			return err
		}

		err = moveFileToYearDir(path, year)
		if err != nil {
			return err
		}
	}
	return nil
}

func getGitCommitYear(filePath string) (string, error) {
	cmd := exec.Command("git", "log", "-1", "--format=%ad", "--date=format:%Y", filePath)
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	year := strings.TrimSpace(string(output))
	return year, nil
}

func moveFileToYearDir(filePath, year string) error {
	dirPath := filepath.Join(filepath.Dir(filePath), year)
	err := os.MkdirAll(dirPath, os.ModePerm)
	if err != nil {
		return err
	}

	newFilePath := filepath.Join(dirPath, filepath.Base(filePath))
	err = os.Rename(filePath, newFilePath)
	if err != nil {
		return err
	}

	fmt.Printf("Moved %s to %s\n", filePath, newFilePath)
	return nil
}
