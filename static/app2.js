document.body.style.backgroundColor = "white";
document.body.style.margin = "50px 10px 10px 50px";
$(".custom-file-input").on("change", function() {
			  var img = $(this).val().split("\\").pop();
			  $(this).siblings(".custom-file-label").addClass("selected").html(img);
			});