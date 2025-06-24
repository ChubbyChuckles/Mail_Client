@echo off
cd C:\Users\Chuck\Desktop\CR_AI_Engineering\Projekte\Mail_Client_Test
echo. > file_contents.txt
echo === pytest.ini === >> file_contents.txt
type pytest.ini >> file_contents.txt
echo. >> file_contents.txt
echo === tests/test_exchange.py === >> file_contents.txt
type tests\test_exchange.py >> file_contents.txt
echo. >> file_contents.txt
echo === tests/test_portfolio.py === >> file_contents.txt
type tests\test_portfolio.py >> file_contents.txt
echo. >> file_contents.txt
echo === tests/test_utils.py === >> file_contents.txt
type tests\test_utils.py >> file_contents.txt