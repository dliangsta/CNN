all: src/**/*.java
	mkdir -p ./bin
	javac -d ./bin ./src/**/*.java

test:
	java -cp ./bin driver.Driver

clean:
	rm -rf ./bin

