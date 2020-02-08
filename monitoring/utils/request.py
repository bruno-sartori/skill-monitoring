import os
import requests
from .logger import LOG

defaultHeaders = { 'Authorization': 'JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MSwibG9naW4iOiJicnVub3NhcnRvcmkud2VibWFzdGVyQGdtYWlsLmNvbSIsImNsaWVudCI6MSwiZGF0ZSI6IjIwMTktMDgtMTZUMDE6MzA6MTMuMDM0WiJ9.9z14DPcEAGUEE1MF27-BuBsRZHUav0jbNBy-zhYwVlU' }

class Request:
	def get(route, params):
		try:
			url = os.path.join(os.environ['CORE_HOST'], route)
			response = requests.get(url, headers=defaultHeaders, verify=False)
			LOG.info(response)
			return response
		except requests.exceptions.RequestException as error:
			LOG.error(error)

	def post(route, data, params):
		try:
			url = os.path.join(os.environ['CORE_HOST'], route)

			response = requests.post(url,
				data=data,
				headers=defaultHeaders,
				verify=False)

			LOG.info(response)
			return response
		except requests.exceptions.RequestException as error:
			LOG.error(error)

	def put(route, data, params):
		try:
			url = os.path.join(os.environ['CORE_HOST'], route)

			response = requests.put(url,
                            data=data,
                            headers=defaultHeaders,
                            verify=False)

			LOG.info(response)
			return response
		except requests.exceptions.RequestException as error:
			LOG.error(error)

	def delete(route, params):
		try:
			url = os.path.join(os.environ['CORE_HOST'], route)

			response = requests.delete(url,
                            headers=defaultHeaders,
                            verify=False)

			LOG.info(response)
			return response
		except requests.exceptions.RequestException as error:
			LOG.error(error)
