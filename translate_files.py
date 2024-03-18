import os
import glob
import deepl

class DocumentTranslator:
    def __init__(self, api_key):
        self.api_key = api_key

    def translate_document(self, input_path, output_path, target_lang="EN-US"):
        try:
            translator = deepl.Translator(self.api_key)
            with open(input_path, "rb") as in_file, open(output_path, "wb") as out_file:
                translator.translate_document(
                    in_file,
                    out_file,
                    target_lang=target_lang,
                )
            print(f"Translation completed: {output_path}")
        except deepl.DocumentTranslationException as error:
            doc_id = error.document_handle.id
            doc_key = error.document_handle.key
            print(f"Error after uploading {error}, id: {doc_id} key: {doc_key}")
        except deepl.DeepLException as error:
            print(error)
        except Exception as error:
            print(f"Error: {error}")

    def translate_documents_in_directory(self, directory, output_directory, target_lang="DE"):
        pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            output_path = os.path.join(output_directory, f"Translated_{filename}")
            self.translate_document(pdf_file, output_path, target_lang)



if __name__ == "__main__":
    api_key = os.environ['DEEPL_API_KEY']
    translator = DocumentTranslator(api_key)
    input_directory = "./data"
    output_directory = "translated_data"

    translator.translate_documents_in_directory(input_directory, output_directory, target_lang="EN-US")
