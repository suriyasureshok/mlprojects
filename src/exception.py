"""
This File creates a custom exception.
"""
import sys
import logging

def error_message_details(error, error_detail:sys):
    """
    This Function helps to get and return the error message details and the error detail from the sys module.
    """
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno,str(error)
    )

    return error_message


class CustomException(Exception):
    """
    This class helps to create a custom exception.
    """
    def __init__(self, error_message, error_detail:sys):
        """
        This method overrides the constructor of the parent Exception class.
        """
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e,sys)
    
"""
This is a code that was used to test the custom exception.
""" 