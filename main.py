"""Main script."""

if __name__ == "__main__":
    SUB = "changemyview"
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2020, 12, 1)

START_DATE_TMSTP, START_DATE_INT = pushshift.convert_datetime(START_DATE)
END_DATE_TMSTP, END_DATE_INT = pushshift.convert_datetime(END_DATE)

df_titles = pushshift.get_all_titles(sub="changemyview",
                                      start_timestamp=datetime(2015, 1, 1),
                                      end_timestamp=datetime(2020, 12, 1),
                                      token=access_token
                                      path_save=
                                     )