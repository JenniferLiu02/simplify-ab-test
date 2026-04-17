#!/usr/bin/env /opt/homebrew/bin/python3.12
"""
02c_import_gpt_data.py
Fixes format issues in GPT-generated data and saves to llm_wide.csv / llm_long.csv.
Fixes applied:
  - used_ai: 0/1  →  No/Yes
  - AC: T_ column names  →  C_  column names
  - AC: adds data_source = "llm"
  - AC: re-numbers person_id to 1101-1153
  - Both: adds empty B/C columns for the other page
"""
import io, os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTCOMES  = ["signup", "useful", "regular", "recommend"]

# ──────────────────────────────────────────────────────────────────────────────
AB_RAW = """person_id,group,data_source,gender,edu_level,year_in_school,age,job_status,field,difficulty,time_per_week,used_ai,heard_simplify,A_signup,A_useful,A_regular,A_recommend,B_signup,B_useful,B_regular,B_recommend
1001,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1002,AB,llm,Male,Master's student,Master's student (1st year),24,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,3,3,4.0,3.0,4.0,3.0
1003,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for full-time jobs,Computer Science / Engineering,Somewhat difficult,0-2 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1004,AB,llm,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
1005,AB,llm,Male,Master's student,Master's student (2nd year or above),25,Actively applying for internships,Information Systems / Data Science,Extremely difficult,More than 8 hours,1,Yes,5,4.0,5,4,4.0,5.0,5.0,5.0
1006,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1007,AB,llm,Female,Master's student,Master's student (1st year),24,Exploring opportunities but not applying yet,Design,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1008,AB,llm,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1009,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,3,3.0,3,3,4.0,3.0,4.0,3.0
1010,AB,llm,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
1011,AB,llm,Female,Master's student,Master's student (1st year),24,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1012,AB,llm,Male,Master's student,Master's student (1st year),31,Actively applying for full-time jobs,Business,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1013,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1014,AB,llm,Female,Master's student,Master's student (2nd year or above),24,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
1015,AB,llm,Male,Master's student,PhD year 3+,28,Exploring opportunities but not applying yet,Social Sciences,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1016,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1017,AB,llm,Male,Master's student,Master's student (1st year),26,Actively applying for full-time jobs,Computer Science / Engineering,Somewhat difficult,6-8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1018,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1019,AB,llm,Female,Master's student,Master's student (1st year),24,Not currently looking for jobs,Information Systems / Data Science,Extremely difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
1020,AB,llm,Male,Master's student,Master's student (2nd year or above),25,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,More than 8 hours,1,Yes,5,4.0,5,4,4.0,5.0,5.0,5.0
1021,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1022,AB,llm,Female,Master's student,Master's student (1st year),21,Actively applying for internships,Information Systems / Data Science,Neither easy nor difficult,3-5 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1023,AB,llm,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,3,3.0,3,3,4.0,3.0,4.0,3.0
1024,AB,llm,Female,Master's student,Master's student (1st year),24,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,0-2 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
1025,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1026,AB,llm,Male,Master's student,Master's student (1st year),31,Actively applying for full-time jobs,Business,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1027,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1028,AB,llm,Female,Master's student,Master's student (2nd year or above),24,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
1029,AB,llm,Male,Master's student,PhD year 3+,28,Exploring opportunities but not applying yet,Social Sciences,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1030,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1031,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1032,AB,llm,Male,Master's student,Master's student (1st year),24,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,3,3,4.0,3.0,4.0,3.0
1033,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for full-time jobs,Computer Science / Engineering,Somewhat difficult,0-2 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1034,AB,llm,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
1035,AB,llm,Male,Master's student,Master's student (2nd year or above),25,Actively applying for internships,Information Systems / Data Science,Extremely difficult,More than 8 hours,1,Yes,5,4.0,5,4,4.0,5.0,5.0,5.0
1036,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1037,AB,llm,Female,Master's student,Master's student (1st year),24,Exploring opportunities but not applying yet,Design,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1038,AB,llm,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1039,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,3,3.0,3,3,4.0,3.0,4.0,3.0
1040,AB,llm,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
1041,AB,llm,Female,Master's student,Master's student (1st year),24,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1042,AB,llm,Male,Master's student,Master's student (1st year),31,Actively applying for full-time jobs,Business,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1043,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1044,AB,llm,Female,Master's student,Master's student (2nd year or above),24,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
1045,AB,llm,Male,Master's student,PhD year 3+,28,Exploring opportunities but not applying yet,Social Sciences,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1046,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1047,AB,llm,Male,Master's student,Master's student (1st year),26,Actively applying for full-time jobs,Computer Science / Engineering,Somewhat difficult,6-8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1048,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1049,AB,llm,Female,Master's student,Master's student (1st year),24,Not currently looking for jobs,Information Systems / Data Science,Extremely difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
1050,AB,llm,Male,Master's student,Master's student (2nd year or above),25,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,More than 8 hours,1,Yes,5,4.0,5,4,4.0,5.0,5.0,5.0
1051,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1052,AB,llm,Female,Master's student,Master's student (1st year),21,Actively applying for internships,Information Systems / Data Science,Neither easy nor difficult,3-5 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1053,AB,llm,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,3,3.0,3,3,4.0,3.0,4.0,3.0
1054,AB,llm,Female,Master's student,Master's student (1st year),24,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,0-2 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
1055,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1056,AB,llm,Male,Master's student,Master's student (1st year),31,Actively applying for full-time jobs,Business,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1057,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1058,AB,llm,Female,Master's student,Master's student (2nd year or above),24,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
1059,AB,llm,Male,Master's student,PhD year 3+,28,Exploring opportunities but not applying yet,Social Sciences,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1060,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1061,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1062,AB,llm,Male,Master's student,Master's student (1st year),24,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,3,3,4.0,3.0,4.0,3.0
1063,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for full-time jobs,Computer Science / Engineering,Somewhat difficult,0-2 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1064,AB,llm,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
1065,AB,llm,Male,Master's student,Master's student (2nd year or above),25,Actively applying for internships,Information Systems / Data Science,Extremely difficult,More than 8 hours,1,Yes,5,4.0,5,4,4.0,5.0,5.0,5.0
1066,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1067,AB,llm,Female,Master's student,Master's student (1st year),24,Exploring opportunities but not applying yet,Design,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1068,AB,llm,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1069,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,3,3.0,3,3,4.0,3.0,4.0,3.0
1070,AB,llm,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
1071,AB,llm,Female,Master's student,Master's student (1st year),24,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1072,AB,llm,Male,Master's student,Master's student (1st year),31,Actively applying for full-time jobs,Business,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1073,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1074,AB,llm,Female,Master's student,Master's student (2nd year or above),24,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
1075,AB,llm,Male,Master's student,PhD year 3+,28,Exploring opportunities but not applying yet,Social Sciences,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1076,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1077,AB,llm,Male,Master's student,Master's student (1st year),26,Actively applying for full-time jobs,Computer Science / Engineering,Somewhat difficult,6-8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1078,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1079,AB,llm,Female,Master's student,Master's student (1st year),24,Not currently looking for jobs,Information Systems / Data Science,Extremely difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
1080,AB,llm,Male,Master's student,Master's student (2nd year or above),25,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,More than 8 hours,1,Yes,5,4.0,5,4,4.0,5.0,5.0,5.0
1081,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1082,AB,llm,Female,Master's student,Master's student (1st year),21,Actively applying for internships,Information Systems / Data Science,Neither easy nor difficult,3-5 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1083,AB,llm,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,3,3.0,3,3,4.0,3.0,4.0,3.0
1084,AB,llm,Female,Master's student,Master's student (1st year),24,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,0-2 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
1085,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1086,AB,llm,Male,Master's student,Master's student (1st year),31,Actively applying for full-time jobs,Business,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1087,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1088,AB,llm,Female,Master's student,Master's student (2nd year or above),24,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
1089,AB,llm,Male,Master's student,PhD year 3+,28,Exploring opportunities but not applying yet,Social Sciences,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1090,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1091,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
1092,AB,llm,Male,Master's student,Master's student (1st year),24,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,3,3,4.0,3.0,4.0,3.0
1093,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for full-time jobs,Computer Science / Engineering,Somewhat difficult,0-2 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
1094,AB,llm,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
1095,AB,llm,Male,Master's student,Master's student (2nd year or above),25,Actively applying for internships,Information Systems / Data Science,Extremely difficult,More than 8 hours,1,Yes,5,4.0,5,4,4.0,5.0,5.0,5.0
1096,AB,llm,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1097,AB,llm,Female,Master's student,Master's student (1st year),24,Exploring opportunities but not applying yet,Design,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
1098,AB,llm,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Information Systems / Data Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
1099,AB,llm,Female,Master's student,Master's student (1st year),22,Actively applying for internships,Information Systems / Data Science,Extremely difficult,6-8 hours,1,Yes,3,3.0,3,3,4.0,3.0,4.0,3.0
1100,AB,llm,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Information Systems / Data Science,Somewhat difficult,3-5 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0"""

# ──────────────────────────────────────────────────────────────────────────────
AC_RAW = """person_id,group,gender,edu_level,year_in_school,age,job_status,field,difficulty,time_per_week,used_ai,heard_simplify,A_signup,A_useful,A_regular,A_recommend,T_signup,T_useful,T_regular,T_recommend
48,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
49,AC,Male,Master's student,Master's student (1st year),24,Actively applying for internships,Data Analytic for Science,Extremely difficult,6-8 hours,1,Yes,3,3.0,3,3,4.0,3.0,4.0,3.0
50,AC,Female,Master's student,Master's student (1st year),22,Actively applying for full-time jobs,Computer Science / Engineering,Somewhat difficult,0-2 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
51,AC,Female,Master's student,Master's student (2nd year or above),25,Not currently looking for jobs,Business,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
52,AC,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
53,AC,Female,Master's student,Master's student (1st year),23,Exploring opportunities but not applying yet,Design,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
54,AC,Male,Master's student,Master's student (1st year),26,Actively applying for full-time jobs,Computer Science / Engineering,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
55,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
56,AC,Female,Master's student,Undergraduate 2nd year,22,Actively applying for internships,Arts,Somewhat difficult,3-5 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
57,AC,Male,Master's student,Master's student (1st year),24,Actively applying for internships,Data Analytic for Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
58,AC,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Business,Somewhat difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
59,AC,Female,Master's student,Master's student (2nd year or above),25,Actively applying for internships,Data Analytic for Science,Somewhat difficult,6-8 hours,1,Yes,5,4.0,5,4,4.0,5.0,5.0,5.0
60,AC,Male,Master's student,Master's student (1st year),30,Actively applying for full-time jobs,Business,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
61,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
62,AC,Male,Master's student,Master's student (1st year),29,Actively applying for internships,Computer Science / Engineering,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
63,AC,Female,Master's student,Master's student (1st year),23,Exploring opportunities but not applying yet,Design,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
64,AC,Female,Master's student,Master's student (1st year),24,Actively applying for internships,Data Analytic for Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
65,AC,Male,Master's student,Master's student (2nd year or above),28,Not currently looking for jobs,Business,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
66,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
67,AC,Male,Master's student,Master's student (1st year),31,Actively applying for full-time jobs,Computer Science / Engineering,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
68,AC,Female,Master's student,Master's student (1st year),24,Actively applying for internships,Data Analytic for Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
69,AC,Female,Master's student,Undergraduate 2nd year,22,Actively applying for internships,Arts,Somewhat difficult,0-2 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
70,AC,Male,Master's student,Master's student (1st year),25,Actively applying for internships,Data Analytic for Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
71,AC,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Business,Somewhat difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
72,AC,Male,Master's student,Master's student (2nd year or above),29,Actively applying for full-time jobs,Computer Science / Engineering,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
73,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
74,AC,Male,Master's student,Master's student (1st year),30,Exploring opportunities but not applying yet,Design,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
75,AC,Female,Master's student,Master's student (1st year),24,Actively applying for internships,Data Analytic for Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
76,AC,Female,Master's student,Master's student (2nd year or above),28,Not currently looking for jobs,Business,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
77,AC,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
78,AC,Female,Master's student,Master's student (1st year),22,Actively applying for internships,Arts,Somewhat difficult,3-5 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
79,AC,Male,Master's student,Master's student (1st year),24,Actively applying for full-time jobs,Computer Science / Engineering,Somewhat difficult,6-8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
80,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
81,AC,Female,Master's student,Master's student (1st year),24,Not currently looking for jobs,Business,Somewhat difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
82,AC,Male,Master's student,Master's student (2nd year or above),29,Actively applying for internships,Computer Science / Engineering,Extremely difficult,More than 8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
83,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
84,AC,Male,Master's student,Master's student (1st year),31,Actively applying for full-time jobs,Computer Science / Engineering,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
85,AC,Female,Master's student,Master's student (1st year),23,Exploring opportunities but not applying yet,Design,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
86,AC,Female,Master's student,Master's student (1st year),24,Actively applying for internships,Data Analytic for Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
87,AC,Male,Master's student,Master's student (2nd year or above),28,Not currently looking for jobs,Business,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
88,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
89,AC,Male,Master's student,Master's student (1st year),30,Actively applying for full-time jobs,Business,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
90,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
91,AC,Female,Master's student,Undergraduate 2nd year,22,Actively applying for internships,Arts,Somewhat difficult,0-2 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0
92,AC,Male,Master's student,Master's student (1st year),24,Actively applying for internships,Data Analytic for Science,Somewhat difficult,3-5 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
93,AC,Female,Master's student,Master's student (1st year),23,Not currently looking for jobs,Business,Somewhat difficult,3-5 hours,0,No,2,2.0,1,2,2.0,2.0,2.0,2.0
94,AC,Male,Master's student,Master's student (2nd year or above),29,Actively applying for full-time jobs,Computer Science / Engineering,Extremely difficult,More than 8 hours,1,Yes,3,2.0,3,3,3.0,3.0,3.0,3.0
95,AC,Female,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Somewhat difficult,6-8 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
96,AC,Male,Master's student,Master's student (1st year),31,Exploring opportunities but not applying yet,Design,Neither easy nor difficult,3-5 hours,1,Not sure,3,3.0,3,3,3.0,3.0,3.0,3.0
97,AC,Female,Master's student,Master's student (1st year),24,Actively applying for internships,Data Analytic for Science,Somewhat difficult,0-2 hours,1,Yes,4,3.0,4,4,4.0,4.0,4.0,4.0
98,AC,Female,Master's student,Master's student (2nd year or above),28,Not currently looking for jobs,Business,Somewhat difficult,3-5 hours,0,No,2,2.0,2,2,2.0,2.0,2.0,2.0
99,AC,Male,Master's student,Master's student (1st year),23,Actively applying for internships,Data Analytic for Science,Extremely difficult,6-8 hours,1,Yes,4,3.0,4,3,4.0,4.0,4.0,4.0
100,AC,Female,Master's student,Master's student (1st year),22,Actively applying for internships,Arts,Somewhat difficult,3-5 hours,0,No,2,1.0,2,2,2.0,2.0,2.0,2.0"""

# ──────────────────────────────────────────────────────────────────────────────
# PARSE
# ──────────────────────────────────────────────────────────────────────────────
import io
df_ab = pd.read_csv(io.StringIO(AB_RAW))
df_ac = pd.read_csv(io.StringIO(AC_RAW))

# ── Fix 1: used_ai  0/1 → No/Yes
ai_map = {0: "No", 1: "Yes", "0": "No", "1": "Yes"}
df_ab["used_ai"] = df_ab["used_ai"].map(ai_map).fillna(df_ab["used_ai"])
df_ac["used_ai"] = df_ac["used_ai"].map(ai_map).fillna(df_ac["used_ai"])

# ── Fix 2: AC – rename T_ → C_, add data_source, re-number person_id
df_ac.rename(columns={
    "T_signup":    "C_signup",
    "T_useful":    "C_useful",
    "T_regular":   "C_regular",
    "T_recommend": "C_recommend",
}, inplace=True)
df_ac["data_source"] = "llm"
df_ac["person_id"]   = range(1101, 1101 + len(df_ac))

# ── Fix 3: add empty opposite-page columns
for o in OUTCOMES:
    df_ab[f"C_{o}"] = None
    df_ac[f"B_{o}"] = None

# ── Assemble final column order
base_cols  = ["person_id","group","data_source","gender","edu_level","year_in_school",
              "age","job_status","field","difficulty","time_per_week","used_ai","heard_simplify"]
score_cols = ([f"A_{o}" for o in OUTCOMES]
             + [f"B_{o}" for o in OUTCOMES]
             + [f"C_{o}" for o in OUTCOMES])

df_llm = pd.concat([df_ab, df_ac], ignore_index=True)
df_llm = df_llm.reindex(columns=base_cols + score_cols)

# Clip scores 0-5
for col in score_cols:
    df_llm[col] = pd.to_numeric(df_llm[col], errors="coerce")
    if df_llm[col].notna().any():
        df_llm[col] = df_llm[col].clip(0, 5).round()

# ── Print sanity check
ab = df_llm[df_llm.group == "AB"]
ac = df_llm[df_llm.group == "AC"]
print(f"AB rows: {len(ab)}  |  AC rows: {len(ac)}")
print(f"used_ai values (AB): {df_llm[df_llm.group=='AB']['used_ai'].value_counts().to_dict()}")
print(f"used_ai values (AC): {df_llm[df_llm.group=='AC']['used_ai'].value_counts().to_dict()}")

b_diff = (ab["B_signup"] - ab["A_signup"]).mean()
c_diff = (ac["C_signup"] - ac["A_signup"]).mean()
print(f"\nMean B-A signup diff (LLM): {b_diff:+.3f}  (human: +0.000)")
print(f"Mean C-A signup diff (LLM): {c_diff:+.3f}  (human: -0.250)")

# ── Save wide
wide_path = os.path.join(DATA_DIR, "llm_wide.csv")
df_llm.to_csv(wide_path, index=False)
print(f"\nSaved: llm_wide.csv  ({len(df_llm)} rows)")

# ── Build and save long
def wide_to_long(df_wide):
    base = ["person_id","group","data_source","gender","edu_level","year_in_school",
            "age","job_status","field","difficulty","time_per_week","used_ai","heard_simplify"]
    rows = []
    for _, r in df_wide.iterrows():
        b = {c: r.get(c) for c in base}
        sp = "B" if r["group"] == "AB" else "C"
        rows.append({**b, "page":"A", "is_treatment":0,
                     "signup":r["A_signup"], "useful":r["A_useful"],
                     "regular":r["A_regular"], "recommend":r["A_recommend"]})
        rows.append({**b, "page":sp, "is_treatment":1,
                     "signup":r[f"{sp}_signup"], "useful":r[f"{sp}_useful"],
                     "regular":r[f"{sp}_regular"], "recommend":r[f"{sp}_recommend"]})
    return pd.DataFrame(rows)

df_long = wide_to_long(df_llm)
long_path = os.path.join(DATA_DIR, "llm_long.csv")
df_long.to_csv(long_path, index=False)
print(f"Saved: llm_long.csv  ({len(df_long)} rows)")
print("\nDone. Run 03_analysis.py next.")
