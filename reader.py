import sign_language_translator as slt

# download dataset or models (if you need them separate)
# (by default, dataset is auto-downloaded within the install directory)
# slt.set_resource_dir("path/to/sign-language-datasets") # optional. Helps when data is synced with cloud

# slt.utils.download("path", "url") # optional
# slt.utils.download_resource(".*.json") # optional

print("All available models:")
print(list(slt.ModelCodes)) # from slt.config.enums
# print(list(slt.TextLanguageCodes))
# print(list(slt.SignLanguageCodes))