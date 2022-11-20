from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, HTTPServer

file = ""

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        file_content = self.rfile.read(content_length)

        # Save the data
        print("writing the data to" , file)
        f = open(file, 'wb')
        f.write(file_content)
        f.close()




        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        message = "done"
        self.wfile.write(bytes(message, "utf8"))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="path of the file to save the uploaded file to",
    )
    args = parser.parse_args()
    file = args.file

    with HTTPServer(('', 1234), handler) as server:
        server.serve_forever()
